import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 가중치 초기화 함수
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 더 안정적인 초기화
                gain = 1.0  # gain 값을 줄임
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 공통 특징 추출 레이어 - 단순화된 구조
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, action_dim)
        )
        
        # 액터의 표준편차 파라미터
        self.actor_direction_std = nn.Parameter(torch.ones(1) * 1.0)  # 더 작은 초기값
        
        # Temperature 파라미터 추가 (클리핑 적용)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # 초기값 1.0으로 설정
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        
        # 가중치 초기화 적용
        self.apply(init_weights)
        
    def forward(self, state):
        # 특징 추출
        features = self.feature_extraction(state)
        
        # 액터: 행동 분포
        action_logits = self.actor_direction(features)
        
        # Temperature scaling 적용
        scaled_logits = action_logits / self.temperature
        
        # 직접적인 softmax 적용
        action_probs = F.softmax(scaled_logits, dim=-1)
        
        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return value, action_probs, action_logits