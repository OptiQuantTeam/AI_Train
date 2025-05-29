import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PriceNormalization(nn.Module):
    def __init__(self, num_features, window_size=20):
        super(PriceNormalization, self).__init__()
        self.window_size = window_size
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6  # 더 큰 epsilon 값 사용

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps  # epsilon 추가
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
            x_norm = (x - mean) / std
        else:
            x_norm = (x - self.running_mean) / (self.running_std + self.eps)
        return torch.clamp(x_norm, -3, 3)

class VolumeNormalization(nn.Module):
    def __init__(self, num_features, window_size=20):
        super(VolumeNormalization, self).__init__()
        self.window_size = window_size
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6

    def forward(self, x):
        # 음수 값 처리
        x = torch.abs(x)
        
        # 로그 변환 (0 값 처리)
        x = torch.log1p(x + self.eps)
        
        if self.training:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps
            
            # NaN 체크 및 처리
            mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
            std = torch.where(torch.isnan(std), torch.ones_like(std), std)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
            
            x_norm = (x - mean) / std
        else:
            x_norm = (x - self.running_mean) / (self.running_std + self.eps)
        
        # NaN 체크 및 처리
        x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
        
        return torch.clamp(x_norm, -3, 3)

class TechnicalIndicatorNormalization(nn.Module):
    def __init__(self, num_features):
        super(TechnicalIndicatorNormalization, self).__init__()
        self.register_buffer('min_val', torch.zeros(num_features))
        self.register_buffer('max_val', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6  # 더 큰 epsilon 값 사용

    def forward(self, x):
        if self.training:
            min_val = x.min(dim=0)[0]
            max_val = x.max(dim=0)[0]
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val
            x_norm = (x - min_val) / (max_val - min_val + self.eps)
        else:
            x_norm = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        return torch.clamp(x_norm, 0, 1)

class SignalNormalization(nn.Module):
    def forward(self, x):
        return torch.clamp(x, -1, 1)
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 가중치 초기화 함수
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 더 큰 gain 값으로 초기화하여 초기 탐색을 촉진
                gain = 2.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 약간의 양의 편향 추가

        #self.price_norm = PriceNormalization(2)
        self.volume_norm = VolumeNormalization(1)
        self.technical_norm = TechnicalIndicatorNormalization(4)
        self.signal_norm = SignalNormalization()
        
        # 공통 특징 추출 레이어 - 단순화된 구조
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 액터의 표준편차 파라미터
        self.actor_direction_std = nn.Parameter(torch.ones(1) * 1.0)  # 더 작은 초기값
        
        # Temperature 파라미터 추가 (클리핑 적용)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # 초기값 1.0으로 설정
        
        self.critic = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
        # 가중치 초기화 적용
        #self.apply(init_weights)
        
    def forward(self, state):
        # 입력 상태를 각 그룹별로 분리
        technical_data = state[:, [2,4,5,6]]  # 인덱스 17,18,19 (기술적 지표)
        signal_data = state[:, [0,1,3,7]]  # 인덱스 24,26,29 (신호 지표)
        
        # 각 그룹별 정규화
        #normalized_price = self.price_norm(price_data)
        #normalized_volume = self.volume_norm(volume_data)
        normalized_technical = self.technical_norm(technical_data)
        normalized_signal = self.signal_norm(signal_data)
        
        # 정규화된 데이터 결합
        normalized_state = torch.cat([
            normalized_technical,
            normalized_signal
        ], dim=1)
        #print(f'normalized_state values: {normalized_state}')
        # 특징 추출
        features = self.feature_extraction(normalized_state)
        
        # 액터: 행동 분포
        action_logits = self.actor_direction(features)
        
        # Temperature scaling 적용
        scaled_logits = action_logits / self.temperature
        
        # 직접적인 softmax 적용
        action_probs = F.softmax(scaled_logits, dim=-1)
        
        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return value, action_probs, action_logits