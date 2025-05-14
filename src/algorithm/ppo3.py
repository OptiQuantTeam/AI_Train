import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import datetime
import network.actorcritic as AC
import network.indicator as ID
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class PPO3:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        model_name=None,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        batch_size=32,
        kl_target=0.01,  # KL 발산 목표값
        kl_coef=0.5,     # KL 발산 계수
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = AC.ActorCritic2(state_dim, action_dim).to(device)
        self.indicator_distribution = ID.IndicatorDistribution3(state_dim, action_dim).to(device)
        
        # 액터 옵티마이저
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.memory = deque()
        self.performances = deque()
        self.alpha = 0.7
        # KL 발산 관련 파라미터
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
        state2 = state[:, [7,9,10,11,17]]
        self.actor_critic.eval()
        with torch.no_grad():
            value, action_probs, action_logits = self.actor_critic(state2)
            pi_I = self.indicator_distribution(state)

            # 두 분포의 평균을 사용
            pi = self.alpha * action_probs + (1 - self.alpha) * pi_I
            action_dist = torch.distributions.Categorical(pi)
            action_idx = action_dist.sample()
            action = action_idx.float() - 1.0
            log_prob = action_dist.log_prob(action_idx)

            
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0]
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, success_rate=None):
        if len(self.memory) < self.batch_size:
            return 0
            
        # 메모리에서 데이터 추출
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서로 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        old_log_prob_batch = torch.FloatTensor(np.array(log_prob_batch)).to(self.device)
        old_value_batch = torch.FloatTensor(np.array(value_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        self.actor_critic.train()
        with torch.no_grad():
            next_state_batch = next_state_batch[:, 5:]
            next_state_batch2 = next_state_batch[:, [7,9,10,11,17]]
            next_value = self.actor_critic(next_state_batch2)[0]  # value는 첫 번째 반환값
            next_value = next_value.squeeze()
            
            for r, v, done, next_v in zip(
                reversed(reward_batch),
                reversed(old_value_batch),
                reversed(done_batch),
                reversed(next_value)
            ):
                if done:
                    delta = r - v
                    gae = delta
                else:
                    delta = r + self.gamma * next_v - v
                    gae = delta + self.gamma * 0.95 * gae
                
                returns.insert(0, gae + v)
                advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트
        final_advantage = None
        final_actor_loss = None
        final_critic_loss = None
        final_entropy_loss = None
        final_total_loss = None
        final_kl_divergence = None
        
        for _ in range(self.epochs):
            # 미니배치 생성 (섞지 않고 순차적으로)
            for start_idx in range(0, len(state_batch), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(state_batch))
                idx = range(start_idx, end_idx)
                
                if len(idx) < self.batch_size:
                    break
                
                # 현재 미니배치
                state = state_batch[idx]
                state = state[:, 5:]
                state2 = state[:, [7,9,10,11,17]]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                value, action_probs, action_logits = self.actor_critic(state2)
                pi_I = self.indicator_distribution(state)
                pi = self.alpha * action_probs + (1 - self.alpha) * pi_I
                
                # Categorical 분포에서 액션 샘플링
                action_dist = torch.distributions.Categorical(pi)
                action_idx = action_dist.sample()
                new_action = action_idx.float() - 1.0
                new_log_prob = action_dist.log_prob(action_idx)
                
                # KL 발산 계산
                kl_divergence = (new_log_prob - old_log_prob).mean()
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 핵심 손실 함수들
                
                # 1. Actor Loss - PPO 클리핑 손실
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # KL 발산 페널티 추가
                kl_penalty = self.kl_coef * torch.max(torch.zeros_like(kl_divergence), 
                                                    kl_divergence - self.kl_target)
                
                # 2. Critic Loss - Huber Loss
                value = value.squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                # 3. 엔트로피 손실 (탐색을 위한)
                entropy_loss = -0.01 * action_dist.entropy().mean()

                # 액터 업데이트                
                #_, action_probs, action_logits = self.actor_critic(state2)

                actor_total_loss = actor_loss + kl_penalty + entropy_loss
                
                
                # 전체 손실 함수 (모니터링용)
                total_loss = actor_total_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                final_advantage = advantage.mean().item()
                final_actor_loss = actor_loss.item()
                final_critic_loss = critic_loss.item()
                final_entropy_loss = entropy_loss.item()
                final_total_loss = total_loss.item()
                final_kl_divergence = kl_divergence.item()
                
        # 성능 지표 저장
        self.store_performance((
            final_advantage,
            final_actor_loss,
            final_critic_loss,
            final_entropy_loss,
            final_total_loss,
            final_kl_divergence
        ))
        
        # 메모리 비우기
        self.memory.clear()
        return 1
    
    def save_model(self, path):
        model_state = {
            # 모델 기본 정보
            'model_name': self.model_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            
            # 모델 가중치 및 옵티마이저 상태
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'indicator_distribution_state_dict': self.indicator_distribution.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # 학습 파라미터
            'learning_params': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epochs': self.epochs,
                'lr_actor': self.optimizer.param_groups[0]['lr'],
                'lr_critic': self.optimizer.param_groups[-1]['lr'],
                'batch_size': self.batch_size,
                'device': str(self.device)
            }
        }
        
        # 변환된 데이터 저장
        torch.save(model_state, path)
    
    def save_learning_state(self, info, path):
        learning_state = {
            # 학습 진행 상태
            'training_state': {
                'current_episode': info['training_state']['current_episode'],
                'total_episodes': info['training_state']['total_episodes'],
                'last_step': info['training_state']['last_step'],
                'checkpoint_term': info['training_state']['checkpoint_term']
            },
            
            # 학습 결과
            'training_results': {
                'rewards_history': info['training_results']['rewards_history'],
                'episode_results': info['training_results']['episode_results'],
                'completed_episodes': sum(info['training_results']['episode_results']),
                'win_rate': info['training_results']['win_rate'],
                'profit_rate_history': info['training_results']['profit_rate_history'],
                'all_balance_history': info['training_results']['all_balance_history'],
                'step_num_history': info['training_results']['step_num_history']
            },

            # 환경 정보
            'environment_info': {
                'data_path': info['environment_info']['data_path'],
                'total_data_length': info['environment_info']['total_data_length'],
                'training_period': info['environment_info']['training_period']
            },

            # 세션 정보
            'session_info': {
                'session_type': info['session_info']['session_type'],
                'session_time': info['session_info']['session_time'],
                'log_file': info['session_info']['log_file'],
                'previous_checkpoints': info['session_info']['previous_checkpoints'],
                'previous_episodes': info['session_info']['previous_episodes'],
                'current_session_episodes': info['session_info']['current_session_episodes'],
                'training_sessions': info['session_info']['training_sessions']
            }
        }

        # NumPy 배열과 숫자를 Python 기본 타입으로 변환
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        learning_state = convert_to_serializable(learning_state)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(learning_state, f, indent=4, ensure_ascii=False)
    
    def store_performance(self, performance):
        self.performances.append(performance)
    
    def plot_performance(self, path):
        # 메모리에서 데이터 추출
        advantages = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        total_losses = []
        kl_divergences = []
        
        for performance in self.performances:
            advantage, actor_loss, critic_loss, entropy_loss, total_loss, kl_divergence = performance
            advantages.append(advantage)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_losses.append(entropy_loss)
            total_losses.append(total_loss)
            kl_divergences.append(kl_divergence)
        
        # 그래프 스타일 설정
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Advantage 분포 히스토그램
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(advantages, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', label='Zero Advantage')
        ax1.axvline(x=np.mean(advantages), color='blue', linestyle='--', label='Mean Advantage')
        ax1.set_title('Advantage Distribution')
        ax1.set_xlabel('Advantage')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Advantage 통계 정보 추가
        positive_ratio = sum(1 for x in advantages if x > 0) / len(advantages)
        ax1.text(0.05, 0.95, 
                f'Positive Ratio: {positive_ratio:.2%}\n'
                f'Mean: {np.mean(advantages):.2f}\n'
                f'Std: {np.std(advantages):.2f}',
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Policy/Value Loss 라인 차트
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(actor_losses, label='Policy Loss', color='blue')
        ax2.plot(critic_losses, label='Value Loss', color='red')
        ax2.plot(total_losses, label='Total Loss', color='orange')
        ax2.set_title('Policy, Value and Total Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # 3. Entropy Loss 라인 차트
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(entropy_losses, label='Entropy Loss', color='green')
        ax3.set_title('Entropy Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        
        # 4. KL 발산 라인 차트
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(kl_divergences, label='KL Divergence', color='purple')
        ax4.axhline(y=self.kl_target, color='r', linestyle='--', label='KL Target')
        ax4.set_title('KL Divergence')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('KL Divergence')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
        self.performances.clear()