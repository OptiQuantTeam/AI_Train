import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic2, self).__init__()
        
        # 공통 특징 추출 레이어 (더 깊고 넓은 구조)
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        
        self.actor_direction_std = nn.Parameter(torch.zeros(1))
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(16, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
    def forward(self, state):
        if state.size(0) == 1:
            self.feature_extraction.eval()
            self.actor_direction.eval()
            self.critic.eval()
        else:
            self.feature_extraction.train()
            self.actor_direction.train()
            self.critic.train()
        
        # 상태 벡터에는 이미 기술적 지표들이 포함되어 있음
        features = self.feature_extraction(state)
        
        # 액터: 행동 분포
        action_logits = self.actor_direction(features)
        
        # softmax를 사용하여 확률 계산
        action_probs = F.softmax(action_logits, dim=-1)

        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return value, action_probs, action_logits

