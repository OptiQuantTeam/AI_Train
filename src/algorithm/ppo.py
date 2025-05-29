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

class PPO:
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
        alpha=0.5,
        entropy_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = AC.ActorCritic(state_dim, action_dim).to(device)
        self.indicator_distribution = ID.IndicatorDistribution(state_dim, action_dim).to(device)
        
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
        self.epsilon = epsilon      # 클리핑 파라미터
        self.epsilon_start = 0.2  # 초기 랜덤 행동 확률
        self.epsilon_end = 0.01   # 최소 랜덤 행동 확률
        self.epsilon_decay = 0.995  # 감소율
        self.current_epsilon = self.epsilon_start   # 현재 랜덤 행동 확률
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.memory = deque()
        self.performances = deque()
        self.alpha = alpha
        self.entropy_coef = entropy_coef

        self.test_mode = False
        
        # 전체 에피소드의 loss를 저장할 변수들
        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_entropy_losses = []
        self.episode_total_losses = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_id = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
        state_ac = state[:, [12,16,17,18,19,24,26,29]]
        
        # 디버깅: 입력값 확인
        #print(f'state_ac shape: {state_ac.shape}')
        
        
        self.actor_critic.eval()
        with torch.no_grad():
            value, action_probs, action_logits = self.actor_critic(state_ac)

            pi_I = self.indicator_distribution(state_id)
            #print(f'action_probs: {action_probs}, pi_I: {pi_I}')
            if not self.test_mode:
                # epsilon-greedy 방식으로 랜덤 행동 선택
                if np.random.random() < self.current_epsilon:
                    action = torch.tensor(np.random.choice([-1, 0, 1]))
                    action_idx = action + 1  # -1,0,1 -> 0,1,2로 변환
                    log_prob = torch.log(torch.tensor(1/3)).to(self.device)  # 균등 분포의 로그 확률
                else:
                    # 가장 높은 확률을 가진 행동 선택
                    action_idx = torch.argmax(action_probs)
                    action = action_idx.float() - 1.0
                    log_prob = torch.log(action_probs[0][action_idx])  # 선택된 행동의 로그 확률
                
                # epsilon 값 감소
                self.current_epsilon = max(self.epsilon_end, self.current_epsilon * self.epsilon_decay)
            else:
                #print(f'action_probs: {action_probs}')
                # 테스트 모드에서는 가장 높은 확률을 가진 액션 선택
                ai_idx = torch.argmax(action_probs)
                ai = ai_idx.float() - 1.0
                indicator_idx = torch.argmax(pi_I)
                indicator = indicator_idx.float() - 1.0
                #print(f'action_probs: {action_probs}, pi_I: {pi_I}')
                pi = self.alpha * action_probs + (1 - self.alpha) * pi_I
                action_idx = torch.argmax(pi)
                action = action_idx.float() - 1.0
                log_prob = torch.log(pi[0][action_idx])  # 선택된 행동의 로그 확률

                    
        return (
            action.cpu().numpy(),
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()  # 스칼라 값으로 반환
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self):
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
        
        #print(f'log_prob_batch: {log_prob_batch}')
        #print(f'value_batch: {value_batch}')
        
        
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
            next_state_batch_id = next_state_batch[:, 5:]
            next_state_batch_ac = next_state_batch[:, [12,16,17,18,19,24,26,29]]
            next_value = self.actor_critic(next_state_batch_ac)[0]  # value는 첫 번째 반환값
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
        
        for _ in range(self.epochs):
            # 미니배치 생성 (섞지 않고 순차적으로)
            for start_idx in range(0, len(state_batch), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(state_batch))
                idx = range(start_idx, end_idx)
                
                if len(idx) < self.batch_size:
                    break
                # 현재 미니배치
                state = state_batch[idx]
                state_id = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
                state_ac = state[:, [12,16,17,18,19,24,26,29]]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                value, action_probs, action_logits = self.actor_critic(state_ac)
                pi_I = self.indicator_distribution(state_id)
                pi = self.alpha * action_probs + (1 - self.alpha) * pi_I
                
                # argmax로 행동 선택
                action_idx = torch.argmax(pi, dim=1)
                new_action = action_idx.float() - 1.0
                new_log_prob = torch.log(pi[0][action_idx])                
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 핵심 손실 함수들
                #print(f'ratio: {ratio}, advantage: {advantage}')
                
                # 1. Actor Loss - PPO 클리핑 손실
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                value = torch.clamp(value, -10.0, 10.0)

                # 2. Critic Loss - Huber Loss
                criterion = nn.HuberLoss(delta=1.0)
                critic_loss = criterion(value, return_)
                
                # 3. 엔트로피 손실 (탐색을 위한)
                entropy_loss = -self.entropy_coef * (-(pi * torch.log(pi + 1e-10)).sum(dim=1)).mean()

                actor_total_loss = actor_loss + entropy_loss
                
                
                # 전체 손실 함수 (모니터링용)
                total_loss = actor_total_loss + 0.3 * critic_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                final_advantage = advantage.mean().item()
                final_actor_loss = actor_loss.item()
                final_critic_loss = critic_loss.item()
                final_entropy_loss = entropy_loss.item()
                final_total_loss = total_loss.item()
                
                
        # 성능 지표 저장
        self.store_performance((
            final_advantage,
            final_actor_loss,
            final_critic_loss,
            final_entropy_loss,
            final_total_loss,
        ))
        
        # 메모리 비우기
        self.memory.clear()
        return 1
    
    def save_model(self, path):
        # NumPy 타입을 Python 기본 타입으로 변환
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

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
                'gamma': float(self.gamma),
                'epsilon': float(self.epsilon),
                'epochs': int(self.epochs),
                'lr_actor': float(self.optimizer.param_groups[0]['lr']),
                'lr_critic': float(self.optimizer.param_groups[-1]['lr']),
                'batch_size': int(self.batch_size),
                'alpha': float(self.alpha),
                'entropy_coef': float(self.entropy_coef),
                'device': str(self.device)
            }
        }
        
        # NumPy 타입 변환 적용
        model_state = convert_to_serializable(model_state)
        
        # 모델 저장
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
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        learning_state = convert_to_serializable(learning_state)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(learning_state, f, indent=4, ensure_ascii=False)
    
    def store_performance(self, performance):
        self.episode_actor_losses.append(performance[1])
        self.episode_critic_losses.append(performance[2])
        self.episode_entropy_losses.append(performance[3])
        self.episode_total_losses.append(performance[4])
 
    
    def plot_performance(self, path):
        # 그래프 스타일 설정
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Entropy Loss 라인 차트
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.episode_entropy_losses, color='green', alpha=0.3, label='Entropy Loss')

        # 추세선 추가
        window_size = 100
        if len(self.episode_entropy_losses) >= window_size:
            rolling_mean = np.convolve(self.episode_entropy_losses, np.ones(window_size)/window_size, mode='valid')
            x = np.arange(window_size-1, len(self.episode_entropy_losses))
            ax1.plot(x, rolling_mean, color='darkgreen', linewidth=2, label='Entropy Loss Trend')
        ax1.set_title('Entropy Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 2. Total Loss 라인 차트
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.episode_total_losses, color='purple', alpha=0.3, label='Total Loss')
        # 추세선 추가
        if len(self.episode_total_losses) >= window_size:
            rolling_mean = np.convolve(self.episode_total_losses, np.ones(window_size)/window_size, mode='valid')
            x = np.arange(window_size-1, len(self.episode_total_losses))
            ax2.plot(x, rolling_mean, color='darkviolet', linewidth=2, label='Total Loss Trend')
        ax2.set_title('Total Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # 3. Policy Loss 라인 차트
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(self.episode_actor_losses, color='blue', alpha=0.3, label='Policy Loss')
        # 추세선 추가
        window_size = 100
        if len(self.episode_actor_losses) >= window_size:
            rolling_mean = np.convolve(self.episode_actor_losses, np.ones(window_size)/window_size, mode='valid')
            x = np.arange(window_size-1, len(self.episode_actor_losses))
            ax3.plot(x, rolling_mean, color='darkblue', linewidth=2, label='Policy Loss Trend')
        ax3.set_title('Policy Loss')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        
        # 4. Value Loss 라인 차트
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(self.episode_critic_losses, color='red', alpha=0.3, label='Value Loss')
        # 추세선 추가
        if len(self.episode_critic_losses) >= window_size:
            rolling_mean = np.convolve(self.episode_critic_losses, np.ones(window_size)/window_size, mode='valid')
            x = np.arange(window_size-1, len(self.episode_critic_losses))
            ax4.plot(x, rolling_mean, color='darkred', linewidth=2, label='Value Loss Trend')
        ax4.set_title('Value Loss')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()