import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def plot_balance_history(episode_results, positions, path):
    # 데이터를 4등분
    total_length = len(episode_results)
    chunk_size = total_length // 4
    
    # 한 사진에 4개의 그래프를 그리기 위한 설정
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Balance History by Quarters', fontsize=16, y=0.95)
    
    # 각 분기별 그래프 그리기
    for chunk in range(4):
        # 현재 청크의 시작과 끝 인덱스
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, total_length)
        
        # 현재 청크의 데이터 추출
        chunk_results = episode_results[start_idx:end_idx]
        chunk_positions = positions[start_idx:end_idx]
        
        # 행과 열 인덱스 계산
        row = chunk // 2
        col = chunk % 2
        
        # 선 그래프 그리기
        for i in range(len(chunk_results)-1):
            current_point = (i, chunk_results[i])
            next_point = (i+1, chunk_results[i+1])
            
            if chunk_positions[i] == 1:  # Long 포지션
                axes[row, col].plot([current_point[0], next_point[0]], 
                        [current_point[1], next_point[1]], 
                        color='green', linewidth=2)
            elif chunk_positions[i] == -1:  # Short 포지션
                axes[row, col].plot([current_point[0], next_point[0]], 
                        [current_point[1], next_point[1]], 
                        color='red', linewidth=2)
            else:  # FLAT 포지션
                axes[row, col].plot([current_point[0], next_point[0]], 
                        [current_point[1], next_point[1]], 
                        color='gray', linewidth=1, alpha=0.5)
        
        # 포지션 변경 지점 표시
        for i in range(1, len(chunk_positions)):
            if chunk_positions[i] != chunk_positions[i-1]:
                axes[row, col].scatter(i, chunk_results[i], 
                           color='blue' if chunk_positions[i] == 1 else 'purple' if chunk_positions[i] == -1 else 'gray',
                           s=50)
        
        # 그래프 설정
        quarter_names = ['First', 'Second', 'Third', 'Fourth']
        axes[row, col].set_title(f'{quarter_names[chunk]} Quarter (Steps {start_idx+1} - {end_idx})')
        axes[row, col].set_xlabel('Step')
        axes[row, col].set_ylabel('Balance')
        axes[row, col].grid(True, alpha=0.3)
        
        # 그래프 비율 설정
        axes[row, col].set_aspect('auto')
    
    # 범례 추가 (한 번만)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', label='Long'),
        Line2D([0], [0], color='red', label='Short'),
        Line2D([0], [0], color='gray', label='Flat'),
        Line2D([0], [0], marker='o', color='blue', label='Enter Long', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color='purple', label='Enter Short', markersize=8, linestyle='None')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right')
    
    # 그래프 간격 조정
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_profit_history(episode_results, price_history, path):
    plt.figure(figsize=(15, 10))
    
    # 왼쪽 y축 (보상 그래프)
    ax1 = plt.gca()
    ax1.plot(episode_results, color='blue', label='Balance')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Balance', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 오른쪽 y축 (가격 그래프)
    ax2 = ax1.twinx()
    ax2.plot(price_history, color='red', label='Price', alpha=0.7)
    ax2.set_ylabel('Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 제목 설정
    plt.title('Balance and Price History')
    
    # 범례 추가
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 그리드 추가
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_cumulative_result(cumulative_results, path):
    # 누적합 계산
    cumsum = np.cumsum(cumulative_results)
    
    # 이전 값보다 큰 지점 찾기
    increasing_points = []
    for i in range(1, len(cumsum)):
        if cumsum[i] > cumsum[i-1]:
            increasing_points.append(i)
    
    # 기본 누적합 그래프
    plt.plot(cumsum, label='Cumulative Result')
    '''
    # 증가하는 지점에 마커 추가
    if increasing_points:
        plt.plot(increasing_points, cumsum[increasing_points], 
                'o', color='red', markersize=4, 
                label='Increasing Points', alpha=0.5)
    '''
    plt.title(f'Cumulative Result Graph : [{cumsum[-1]}/{len(cumsum)-1}]')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def plot_cumulative_result_log(cumulative_results, path):
    # 누적합 계산
    cumsum = np.cumsum(cumulative_results)
    
    # 이전 값보다 큰 지점 찾기
    increasing_points = []
    for i in range(1, len(cumsum)):
        if cumsum[i] > cumsum[i-1]:
            increasing_points.append(i)
    
    # 기본 누적합 그래프
    plt.plot(cumsum, label='Cumulative Result')
    
    # 증가하는 지점에 마커 추가
    if increasing_points:
        plt.plot(increasing_points, cumsum[increasing_points], 
                'o', color='red', markersize=4, 
                label='Increasing Points', alpha=0.5)
    
    plt.title(f'Cumulative Result Graph : [{cumsum[-1]}/{len(cumsum)-1}]')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def plot_profit_rate_history(profit_rate_history, path):
    # 색상 조건에 따라 점들의 색상 결정
    colors = []
    for rate in profit_rate_history:
        if rate >= 66:
            colors.append('green')        # 66% 이상: 초록색
        elif rate >= 0:
            colors.append('lightgreen')   # 0~66%: 연두색
        elif rate >= -34:
            colors.append('orange')       # -34~0%: 주황색
        else:
            colors.append('red')          # -34% 이하: 빨간색
    
    plt.scatter(range(len(profit_rate_history)), profit_rate_history, c=colors, marker='o')
    plt.plot(profit_rate_history, color='gray', alpha=0.3)  # 연결선 추가
    plt.title('Profit Rate Graph')
    plt.xlabel('Episode')
    plt.ylabel('Profit Rate')
    plt.savefig(path)
    plt.close()

def plot_reward_history(episode_rewards, path):
    plt.figure(figsize=(15, 8))
    
    plt.plot(episode_rewards)
    plt.title('Reward Graph')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(path)
    plt.close()

def plot_episode_metrics(balance_history, profit_history, price_history, profit_rate_history, actions, balance_profit_rate_history, path=None):
    """
    에피소드별 성능 지표를 시각화하는 함수
    Args:
        balance_history (list): 모든 에피소드의 잔고 기록 리스트
        profit_history (list): 모든 에피소드의 수익 기록 리스트
        price_history (list): 모든 에피소드의 가격 기록 리스트
        profit_rate_history (list): 모든 에피소드의 수익률 기록 리스트
        actions (list): 모든 에피소드의 행동 기록 리스트
        balance_profit_rate_history (list): 모든 에피소드의 잔고 수익률 기록 리스트
        path (str): 그래프를 저장할 경로
    """
    # 그래프 크기 및 레이아웃 설정
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # 1. 가격, 포지션 및 수익률 그래프
    ax1 = plt.subplot(gs[0, :])
    
    # 가격 그래프 (왼쪽 y축)
    ax1.plot(price_history, label='Price', color='navy', alpha=0.7)
    ax1.set_ylabel('Price', fontsize=10, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    
    # 수익률 그래프 (오른쪽 y축)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(balance_profit_rate_history, color='brown', alpha=0.7, label='Profit Rate')
    #ax1_twin.set_ylabel('Profit Rate', fontsize=10, color='brown')
    ax1_twin.tick_params(axis='y', labelcolor='brown')
    
    # 수익률 막대 그래프 (하단)
    ax1_bottom = ax1_twin.twinx()
    #ax1_bottom.spines['right'].set_position(('outward', 60))
    colors = ['green' if rate >= 0 else 'red' for rate in profit_rate_history]
    ax1_bottom.bar(range(len(profit_rate_history)), np.abs(profit_rate_history), 
                  color=colors, alpha=0.5, label='Trade Profit Rate')
    ax1_bottom.set_ylabel('Trade Profit Rate (%)', fontsize=10, color='gray')
    ax1_bottom.tick_params(axis='y', labelcolor='gray')
    
    # 포지션 진입/정리 지점 표시
    for i in range(1, len(actions)):
        # 포지션 진입 지점
        if actions[i] != actions[i-1] and actions[i-1] == 0:  # FLAT에서 LONG/SHORT로 진입
            if actions[i] == 1:  # 롱 진입
                ax1.scatter(i, price_history[i], color='green', marker='^', s=150, alpha=0.4, label='Enter Long' if i == 1 else "")
            elif actions[i] == -1:  # 숏 진입
                ax1.scatter(i, price_history[i], color='red', marker='v', s=150, alpha=0.4, label='Enter Short' if i == 1 else "")
        
        # 포지션 정리 지점
        if actions[i] == 0 and actions[i-1] != 0:  # LONG/SHORT에서 FLAT으로 전환
            profit = profit_history[i] if i < len(profit_history) else 0
            if actions[i-1] == 1:  # 롱 포지션 정리
                if profit > 0:  # 수익 실현
                    ax1.scatter(i, price_history[i], color='blue', marker='o', s=100, alpha=0.8, label='Close Long with Profit' if i == 1 else "")
                else:  # 손실 실현
                    ax1.scatter(i, price_history[i], color='purple', marker='o', s=100, alpha=0.8, label='Close Long with Loss' if i == 1 else "")
            else:  # 숏 포지션 정리
                if profit > 0:  # 수익 실현
                    ax1.scatter(i, price_history[i], color='blue', marker='x', s=100, alpha=0.8, label='Close Short with Profit' if i == 1 else "")
                else:  # 손실 실현
                    ax1.scatter(i, price_history[i], color='purple', marker='x', s=100, alpha=0.8, label='Close Short with Loss' if i == 1 else "")
    
    # 범례 추가
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    handles3, labels3 = ax1_bottom.get_legend_handles_labels()
    ax1.legend(handles + handles2 + handles3, labels + labels2 + labels3, loc='upper left', fontsize='small')
    
    ax1.set_title('Price, Position and Profit Rate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 수익/손실 횟수 그래프
    ax2 = plt.subplot(gs[1, 0])
    profit_count = sum(1 for p in profit_history if p > 0)
    loss_count = sum(1 for p in profit_history if p < 0)
    total_trades = profit_count + loss_count
    win_rate = profit_count / total_trades if total_trades > 0 else 0
    
    ax2.bar(['Profit', 'Loss'], [profit_count, loss_count], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_title(f'Trade Count (Win Rate: {win_rate:.2%})', fontsize=12)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 행동 분포 그래프
    ax3 = plt.subplot(gs[1, 1])
    action_indices = np.array(actions).astype(int) + 1
    action_counts = np.bincount(action_indices, minlength=3)
    action_probs = action_counts / len(action_indices)
    entropy = -np.sum(action_probs * np.log2(action_probs + 1e-8))
    
    action_labels = ['Short', 'Flat', 'Long']
    action_colors = ['red', 'gray', 'green']
    ax3.bar(action_labels, action_probs, color=action_colors)
    ax3.set_title(f'Action Distribution (Entropy: {entropy:.2f})', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 그래프 간격 조정
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 성능 지표 계산
    metrics = {
        'win_rate': win_rate,
        'profit_count': profit_count,
        'loss_count': loss_count,
        'total_trades': total_trades,
        'action_entropy': entropy,
        'action_distribution': dict(zip(action_labels, action_probs))
    }
    
    return metrics

def plot_learning_progress(episode_win_rate, profit_rate_history, episode_rewards, episode_results, path):
    """
    여러 에피소드에 걸친 학습 지표를 시각화하는 함수
    
    Args:
        balance_history (list): 모든 에피소드의 잔고 기록 리스트
        cumulative_results (list): 모든 에피소드의 누적 결과 리스트 
        profit_rate_history (list): 모든 에피소드의 수익률 기록 리스트 %
        episode_rewards (list): 모든 에피소드의 보상 리스트
        path (str): 그래프를 저장할 경로
    """
    num_episodes = len(profit_rate_history)
    
    # 각 에피소드의 지표 계산
    win_rates = []
    
    # 승률 계산 (수익률이 양수일 때)
    for profit_rate in profit_rate_history:
    #for profit_rate in episode_results:
        win_rates.append(1 if profit_rate > 1 else 0)
    
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learning Progress Metrics', fontsize=16)
    
    # 1. 승률 그래프 (누적)
    cumulative_win_rate = np.cumsum(win_rates) / (np.arange(num_episodes) + 1)
    axes[0, 0].plot(cumulative_win_rate)
    axes[0, 0].set_title(f'Cumulative Win Rate : {cumulative_win_rate[-1]:.2%}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Win Rate')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True)
    '''
    step_num = step_num_history
    axes[0, 0].plot(step_num)
    axes[0, 0].set_title(f'Step Number')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Step Number')
    axes[0, 0].grid(True)
    '''
    # 2. 수익률 그래프와 이동평균선 (오른쪽 상단)
    ax2 = axes[0, 1]
    # 수익률 그래프
    x = np.arange(len(profit_rate_history))
    ax2.bar(x, profit_rate_history, color='gray', alpha=0.5, label='Profit Rate')
    
    # 양수/음수 수익률에 따른 색상 구분
    for i, rate in enumerate(profit_rate_history):
        color = 'green' if rate >= 0 else 'red'
        ax2.scatter(i, rate, color=color, s=10, alpha=0.5)
    
    # 이동평균선 추가
    ma_windows = [5, 10, 20]
    colors = ['blue', 'orange', 'purple']
    for i, window in enumerate(ma_windows):
        if len(profit_rate_history) >= window:
            ma = np.convolve(profit_rate_history, np.ones(window)/window, mode='valid')
            x_values = np.arange(window-1, len(profit_rate_history))
            ax2.plot(x_values, ma, color=colors[i % len(colors)], 
                     linewidth=2, label=f'MA-{window}')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Profit Rate with Moving Averages')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Profit Rate (%)')
    ax2.legend(loc='upper left', fontsize='small')
    ax2.grid(True)
    
    # 3. 누적 결과 그래프
    '''
    axes[1, 0].plot(np.cumsum(episode_results))
    axes[1, 0].set_title(f'Cumulative Returns : {np.cumsum(episode_results)[-1]}')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].grid(True)
    '''

    axes[1, 0].plot(episode_win_rate)
    axes[1, 0].set_title(f'Episode win rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].grid(True)
    
    # 4. 보상 변화 그래프
    episode_rewards = np.array(episode_rewards, dtype=float)  # float 타입으로 변환
    #reward_changes = np.diff(episode_rewards)
    reward_changes = episode_rewards
    x_values = np.arange(1, len(reward_changes) + 1)
    
    # 기본 변화 선
    axes[1, 1].plot(x_values, reward_changes, color='lightgray', alpha=0.2, label='Reward Change')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 양수/음수 보상 변화에 따른 색상 구분
    positive_mask = reward_changes > 0
    negative_mask = reward_changes <= 0
    
    if np.any(positive_mask):
        axes[1, 1].scatter(x_values[positive_mask], 
                          reward_changes[positive_mask], 
                          color='green', s=20, alpha=0.6, label='Positive Change')
    
    if np.any(negative_mask):
        axes[1, 1].scatter(x_values[negative_mask], 
                          reward_changes[negative_mask], 
                          color='red', s=20, alpha=0.6, label='Negative Change')
    
    # 이동평균선 추가
    ma_windows = [5, 10]
    colors = ['darkblue', 'darkorange']
    for i, window in enumerate(ma_windows):
        if len(reward_changes) >= window:
            ma = np.convolve(reward_changes, np.ones(window)/window, mode='valid')
            x_values = np.arange(window, len(reward_changes) + 1)
            axes[1, 1].plot(x_values, ma, color=colors[i], 
                           linewidth=2.5, label=f'MA-{window}', alpha=0.9)
    
    axes[1, 1].set_title('Reward Changes by Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward Change')
    axes[1, 1].legend(loc='upper left', fontsize='small')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 통계 정보 추가
    if len(profit_rate_history) > 0:
        avg_profit = np.mean(profit_rate_history)
        win_rate = np.mean([1 if r >= 0 else 0 for r in profit_rate_history])
        max_drawdown = 0
        if len(profit_rate_history) > 1:
            # 누적 수익률 계산 (퍼센트를 소수로 변환)
            returns = np.array(profit_rate_history) / 100
            cumulative_returns = np.cumprod(1 + returns) - 1
            # 최대 낙폭 계산
            max_returns = np.maximum.accumulate(cumulative_returns)
            drawdowns = (max_returns - cumulative_returns) / (1 + max_returns)
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
        
        stat_text = (
            f"Episodes: {len(profit_rate_history)}  |  "
            f"Avg Profit: {avg_profit:.2f}%  |  "
            f"Win Rate: {win_rate:.2%}  |  "
            f"Max Drawdown: {max_drawdown:.2f}%"
        )
        plt.figtext(0.5, 0.01, stat_text, fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
    # 계산된 지표들을 딕셔너리로 반환
    return {
        'cumulative_win_rate': np.sum(episode_results) / num_episodes,
        'total_reward': np.sum(episode_rewards)
    }