import numpy as np
import pandas as pd
import gym
from gym import spaces
from Logger import Logger

# position constant
LONG = 1
SHORT = -1
FLAT = 0

# action constant
BUY = 1
SELL = -1
HOLD = 0

class FuturesEnv(gym.Env):
    def __init__(self, path=None, logger=None):
        '''
        환경의 초기 설정값
        Variables:
            path: 데이터 파일 경로
            logger: 로거 객체
            initial_balance: 초기 자산
            actions: 행동 종류
            initial_leverage: 초기 레버리지
            leverage: 레버리지
            min_leverage: 최소 레버리지
            max_leverage: 최대 레버리지
            leverage_step: 레버리지 조정 단위
            trade_fee: 거래 수수료
            max_steps: 마지막 데이터 위치에서 최대 스텝
            min_steps: PPO 배치 사이즈를 고려한 최소 스텝 수
            profit_target: 수익 목표
            max_position_ratio: 최대 포지션 크기 비율 감소
            stop_loss_threshold: 손절매 임계값 감소
            action_space: 행동 공간
            observation_space: 관찰 공간
            data: 환경 데이터
            success_episodes: 다음 스텝 시작 위치
            last_step: 마지막 스텝(학습 시작 위치)
        '''

        self.path = path
        self.logger = logger
        self.initial_balance = 1000000
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.initial_leverage = 2  # 초기 레버리지
        self.leverage = self.initial_leverage  # 현재 레버리지
        self.min_leverage = 1.0  # 최소 레버리지
        self.max_leverage = 8.0  # 최대 레버리지
        self.leverage_step = 1  # 레버리지 조정 단위
        self.trade_fee = 0.0002
        self.max_steps = 2100
        self.min_steps = 2048  # PPO 배치 사이즈를 고려한 최소 스텝 수
        self.profit_target = 0.1  # 10% 수익 목표
        self.max_position_ratio = 1  # 최대 포지션 크기 비율 감소
        self.stop_loss_threshold = 0.02  # 손절매 임계값 감소
        self.take_profit_threshold = 0.02  # 익절매 임계값 감소
        self.recurrence = 0
        self.test = False
               
        # 레버리지에 따른 손실 제한 관련 파라미터
        self.leverage_loss_limits = {
            8.0: 0.05,  # 2배 레버리지일 때 10%
            4.0: 0.04,  # 1.8배 레버리지일 때 9%
            2.0: 0.03,  # 1.6배 레버리지일 때 8%
            1.0: 0.02,  # 1.4배 레버리지일 때 7%
        }

        self.leverage_profit_limits = {
            8.0: 0.10,  # 2배 레버리지일 때 10%
            4.0: 0.08,  # 1.8배 레버리지일 때 9%
            2.0: 0.06,  # 1.6배 레버리지일 때 8%
            1.0: 0.04,  # 1.4배 레버리지일 때 7%
        }
        
        # 레버리지에 따른 포지션 비중 조절 파라미터
        self.leverage_position_ratios = {
            8.0: 0.4,  # 4배 레버리지일 때 20% 포지션
            4.0: 0.5,  # 3배 레버리지일 때 30% 포지션
            2.0: 0.6,  # 2배 레버리지일 때 40% 포지션
            1.0: 0.7   # 1배 레버리지일 때 50% 포지션
        }
        
        self._load_data()
        
        # 상태 공간 확장 (변동성과 다중 시간프레임 정보 추가)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,),  # 상태 공간 확장
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3, start=-1)

        self.success_episodes = 0
        self.last_step = 100

    def _load_data(self):
        '''
        데이터 로드 및 전처리
        '''
        data = pd.read_csv(self.path)
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, df):
        '''
        데이터 전처리: 결측치 처리 및 정규화
        '''
        # 필요한 컬럼만 선택
        columns = ['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ha_high_diff', 'ha_low_diff', 'ha_body_diff',
                'ema_200', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_middle', 'bb_upper', 'bb_lower',
                'bb_width', 'bb_width_change',
                'MACD', 'MACD_Signal', 'Cross Signal', 'Divergence Signal', 'Trade Signal']
        df = df[columns]

        # 결측치 처리
        df = df.ffill()
        df = df.bfill()
                
        # 이상치 제거 (극단값 제거)
        for column in columns:
            q1 = df[column].quantile(0.01)
            q3 = df[column].quantile(0.99)
            df[column] = df[column].clip(q1, q3)
        
        # 정규화
        '''
        for column in ['Open', 'Close', 'Volume', 'CHG']:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column] - mean) / (std + 1e-8)
        '''

        return df
    
    def _adjust_loss_limit(self):
        '''
        레버리지에 따라 손실 한도를 조정하는 함수
        '''
        # 가장 가까운 레버리지 단계 찾기
        closest_leverage = min(self.leverage_loss_limits.keys(), 
                             key=lambda x: abs(x - self.leverage))
        
        # 해당 레버리지에 맞는 손실 한도 설정
        self.stop_loss_threshold = self.leverage_loss_limits[closest_leverage]
        #self.logger.render(f"현재 레버리지: {self.leverage:.1f}x, 손실 한도: {self.stop_loss_threshold*100:.1f}%")
    
    def _adjust_profit_limit(self):
        '''
        레버리지에 따라 수익 한도를 조정하는 함수
        '''
        closest_leverage = min(self.leverage_profit_limits.keys(), 
                             key=lambda x: abs(x - self.leverage))
        
        # 해당 레버리지에 맞는 수익 한도 설정
        self.take_profit_threshold = self.leverage_profit_limits[closest_leverage]
        #self.logger.render(f"현재 레버리지: {self.leverage:.1f}x, 수익 한도: {self.profit_target*100:.1f}%")

    def _adjust_leverage(self, profit_rate):
        '''
        수익률에 따라 레버리지를 조정하는 함수
        Args:
            profit_rate: 현재 거래의 수익률
        '''
        if profit_rate < 0.001:  # 손실인 경우
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # 연속 손실에 따라 레버리지 감소
            if self.consecutive_losses >= 2:
                self.leverage = max(self.min_leverage, self.leverage - self.leverage_step)
                self._adjust_loss_limit()  # 레버리지 변경 시 손실 한도 재조정
                self._adjust_profit_limit()  # 레버리지 변경 시 수익 한도 재조정
                #self.logger.render(f"연속 손실로 레버리지 감소: {self.leverage:.2f}")
        else:  # 수익인 경우
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # 연속 수익에 따라 레버리지 증가
            if self.consecutive_wins >= 2:
                self.leverage = min(self.max_leverage, self.leverage + self.leverage_step)
                self._adjust_loss_limit()  # 레버리지 변경 시 손실 한도 재조정
                self._adjust_profit_limit()  # 레버리지 변경 시 수익 한도 재조정
                #self.logger.render(f"연속 수익으로 레버리지 증가: {self.leverage:.2f}")

    def _get_position_ratio(self):
        '''
        현재 레버리지에 따른 포지션 비중을 반환하는 함수
        '''
        # 가장 가까운 레버리지 단계 찾기
        closest_leverage = min(self.leverage_position_ratios.keys(), 
                             key=lambda x: abs(x - self.leverage))
        
        # 해당 레버리지에 맞는 포지션 비중 반환
        return self.leverage_position_ratios[closest_leverage]

    def reset(self):
        '''
        환경 초기화 함수
        Variables:
            current_step: 현재 스텝
            last_step: 마지막 스텝
            balance: 현재 자산
            position: 현재 포지션
            action: 현재 행동
            size: 현재 포지션 크기
            position_size: 현재 포지션 크기
            entry_price: 진입 가격
            num: 반복한 스텝
            liquidated: 청산 여부
            clear: 목표 달성 여부
            success: 성공 횟수
            failure: 실패 횟수
            balance_profit_rate: 수익률
            total_trade: 총 거래 횟수
            consecutive_wins: 연속 수익 횟수
            consecutive_losses: 연속 손실 횟수
            leverage: 레버리지 초기화

            reward_history: 현재 에피소드의 step 별 보상 기록
            balance_profit_rate_history: 현재 에피소드의 step 별 자산 수익률 기록
            position_history: 현재 에피소드의 step 별 포지션 기록
            price_history: 현재 에피소드의 step 별 가격 기록
            balance_history: 현재 에피소드의 step 별 자산 기록
            profit_history: 현재 에피소드의 step 별 수익 기록
            profit_rate_history: 현재 에피소드의 step 별 수익률 기록
        Returns:
            state: 다음 상태
        '''

        
        self.current_step = 0

        self.tmp_current = self.current_step
        self.balance = self.initial_balance
        self.position = None
        self.action = HOLD
        self.size = 1e-6
        self.position_size = 0
        self.entry_price = 1
        self.num = 0
        self.liquidated = False
        self.clear = False
        self.success = 0
        self.failure = 0
        self.balance_profit_rate = 0
        self.total_trade = 0
        self.consecutive_wins = 0  # 연속 수익 횟수 초기화
        self.consecutive_losses = 0  # 연속 손실 횟수 초기화
        self.leverage = self.initial_leverage
        self.step_count = 0
        self._adjust_loss_limit()  # 레버리지에 따른 손실 한도 조정
        
        self.reward_history = []
        self.balance_profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.price_history = []
        self.balance_history = []
        self.profit_history = []
        self.profit_rate_history = []
        
        return self._next_observation()
    
    def _next_observation(self):
        '''
        다음 상태 함수 - 기술적 지표 기반 상태 데이터
        Returns:
            state: 다음 상태 (17개 특성)
        '''
       
        # 기술적 지표만 사용 (총 17개)
        state = np.array([
            # 가격 관련 지표
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Volume'],

            self.data.iloc[self.current_step]['ha_open'],
            self.data.iloc[self.current_step]['ha_close'],
            self.data.iloc[self.current_step]['ha_high'],
            self.data.iloc[self.current_step]['ha_low'],
            self.data.iloc[self.current_step]['ha_body'],
            self.data.iloc[self.current_step]['ha_lower_wick'],
            self.data.iloc[self.current_step]['ha_upper_wick'],
            self.data.iloc[self.current_step]['ha_signal'],
            self.data.iloc[self.current_step]['ha_high_diff'],
            self.data.iloc[self.current_step]['ha_low_diff'],
            self.data.iloc[self.current_step]['ha_body_diff'],

            self.data.iloc[self.current_step]['ema_200'],
            self.data.iloc[self.current_step]['ema_200_signal'],
            
            self.data.iloc[self.current_step]['stoch_rsi'],
            self.data.iloc[self.current_step]['stoch_signal'],
            
            self.data.iloc[self.current_step]['bb_middle'],
            self.data.iloc[self.current_step]['bb_upper'],
            self.data.iloc[self.current_step]['bb_lower'],
            self.data.iloc[self.current_step]['bb_width'],
            self.data.iloc[self.current_step]['bb_width_change'],

            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_Signal'],
            self.data.iloc[self.current_step]['Cross Signal'],
            self.data.iloc[self.current_step]['Divergence Signal'],
            self.data.iloc[self.current_step]['Trade Signal']
        ], dtype=np.float32)
        
        return state
        
    def step(self, action):
        '''
        행동 함수
        Args:
            action: 행동 [-1, 0, 1]
            - 1 (LONG): 현재 Long이면 유지, Flat이면 Long 진입, Short이면 Flat으로 청산
            - -1 (SHORT): 현재 Long이면 Flat으로 청산, Flat이면 Short 진입, Short이면 유지
            - 0 (FLAT): 현재 포지션 유지
        Returns:
            state: 다음 상태
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        '''

        current_price = self.data.iloc[self.current_step]['Close']
        done = self.current_step >= len(self.data) - 1
        reward = 0
        do = 0

        # 포지션이 없는 경우
        if self.position is None:
            if action == LONG:
                self.position = LONG
                self.entry_price = current_price
                # 레버리지에 따른 포지션 비중 조절
                trade_ratio = self._get_position_ratio()
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                self.size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = self.size * current_price * self.trade_fee
                self.balance -= entry_cost
                reward = 2.0
                do = self.position
                self.total_trade += 1
                self.logger.render(f"LONG 진입 - 가격: {current_price:.2f}, 크기: {self.size:.2f}, 레버리지: {self.leverage:.1f}x")
            elif action == SHORT:
                self.position = SHORT
                self.entry_price = current_price
                # 레버리지에 따른 포지션 비중 조절
                trade_ratio = self._get_position_ratio()
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                self.size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = self.size * current_price * self.trade_fee
                self.balance -= entry_cost
                reward = 2.0
                do = self.position
                self.total_trade += 1
                self.logger.render(f"SHORT 진입 - 가격: {current_price:.2f}, 크기: {self.size:.2f}, 레버리지: {self.leverage:.1f}x")
            else:
                reward = 0.4
                
        # LONG 포지션인 경우
        elif self.position == LONG:
            if action == SHORT:  # SHORT 또는 HOLD 액션으로 청산
                profit = (current_price - self.entry_price)
                profit_rate = profit / self.entry_price
                self.balance += profit * self.size
                exit_cost = self.size * current_price * self.trade_fee
                self.balance -= exit_cost
                reward = 10 if profit > 0 else -10
                if profit > 0:
                    self.success += 1
                else:
                    self.failure += 1
                do = self.position * 2
                self.logger.render(f"LONG 청산 - 가격: {current_price:.2f}, {'수익' if profit > 0 else '손실'}: {profit:.2f}")
                self.position = None
                self.size = 1e-6
                self._adjust_leverage(profit_rate)
            elif action == FLAT or action == LONG:
                unrealized_profit = (current_price - self.entry_price)
                unrealized_profit_rate = unrealized_profit / self.entry_price
                reward = -0.1
        
        # SHORT 포지션인 경우
        elif self.position == SHORT:
            if action == LONG:  # LONG 또는 HOLD 액션으로 청산
                profit = (self.entry_price - current_price)
                profit_rate = profit / self.entry_price
                self.balance += profit * self.size
                exit_cost = self.size * current_price * self.trade_fee
                self.balance -= exit_cost
                reward = 10 if profit > 0 else -10
                if profit > 0:
                    self.success += 1
                else:
                    self.failure += 1
                do = self.position * 2
                self.logger.render(f"SHORT 청산 - 가격: {current_price:.2f}, {'수익' if profit > 0 else '손실'}: {profit:.2f}")
                self.position = None
                self.size = 1e-6
                self._adjust_leverage(profit_rate)
            elif action == FLAT or action == SHORT:
                unrealized_profit = (self.entry_price - current_price)
                unrealized_profit_rate = unrealized_profit / self.entry_price
                reward = -0.1
        else:
            reward = 0.2

        self.balance_profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
        # 히스토리 기록
        self.price_history.append(current_price)
        self.position_history.append(self.position)
        self.balance_profit_rate_history.append(self.balance_profit_rate * 100)  
        self.profit_history.append(profit if 'profit' in locals() else 0)
        self.balance_history.append(self.balance)
        self.reward_history.append(reward)
        self.profit_rate_history.append(profit_rate * 100 if 'profit_rate' in locals() else 0)
        self.step_count += 1

        if done:
            if self.position != None:
                profit = self.position * (current_price - self.entry_price)
                profit_rate = profit / self.entry_price
                self.balance += profit * self.size
                exit_cost = self.size * current_price * self.trade_fee
                self.balance -= exit_cost
                reward = 10 if profit > 0 else -10

            # 최종 수익률에 따른 보상
            final_profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            
            # 학습 종료 시 다음 학습 step 여부 결정
            if final_profit_rate > 0:
                self.success_episodes += 1
        else:
            self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'trade_success': 0,
            'position': do,
        }

        return self._next_observation(), reward, done, info

    def render(self):
        if self.logger is None:
            return
        
        # Render the environment to the screen                
        profit = float(self.balance - self.initial_balance)
        profit_rate = float((self.balance - self.initial_balance) * 100 / self.initial_balance)
        self.logger.basic(f'학습 마지막 위치: {self.current_step}')
        self.logger.basic(f'Balance: {float(self.balance):.2f}')
        self.logger.basic(f'Profit: {float(profit):.2f}, Profit Rate: {float(profit_rate):.2f}%')
        self.logger.basic(f'Success: {self.success}, Failure: {self.failure}')
        self.logger.basic(f'해당 에피소드 거래 횟수: {self.total_trade}, 누적 에피소드 성공 횟수: {self.success_episodes}')
