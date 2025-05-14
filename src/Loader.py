import torch
from pathlib import Path
import algorithm as algo
import env
from Logger import Logger, LogLevel
import datetime
import numpy as np
import os
import json
from graph import plot_episode_metrics, plot_learning_progress
import glob

class Loader():
    def __init__(self, env_path, env_path_test, further=None, auto=False):
        self.env = env.FuturesEnv11(path=env_path)
        self.test_env = env.FuturesEnv11_test(path=env_path_test)

        if auto:
            self.agent, self.model_info, self.learning_info = self._load_model(further, auto)
            self.logger = Logger(self.agent.model_name, f'logs/{self.agent.model_name}.log', console_level=LogLevel.CRITICAL, file_level=LogLevel.CRITICAL)
            self.env.logger = self.logger
            self.test_env.logger = self.logger
        else:
            self.agent, self.model_info, self.learning_info = self._set_model() if further is None else self._load_model(further, auto)
            console_level, file_level = self._set_log_level()
            self.logger = Logger(self.agent.model_name, f'logs/{self.agent.model_name}.log', console_level=console_level, file_level=file_level)
            self.env.logger = self.logger
            self.test_env.logger = self.logger
            if self.agent is None:
                self.logger.error("모델을 로드할 수 없습니다.")
                exit(1)
            # 모델 정보 출력
            self.logger.render_model_info(self.model_info)
            input('학습을 시작합니다 [Enter]')

    def _set_log_level(self):
        # 로그 레벨 매핑 딕셔너리
        LEVEL_MAP = {
            0: LogLevel.DEBUG,
            1: LogLevel.INFO,
            2: LogLevel.WARNING,
            3: LogLevel.ERROR,
            4: LogLevel.CRITICAL
        }
        
        def get_log_level(prompt, default=1):
            try:
                level = int(input(prompt) or str(default))
                return LEVEL_MAP.get(level, LogLevel.INFO)
            except ValueError:
                return LogLevel.INFO
        
        console_level = get_log_level('콘솔 로그 레벨을 선택해주세요. [0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL] [default: 1]: ')
        file_level = get_log_level('파일 로그 레벨을 선택해주세요. [0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL] [default: 1]: ')
        
        return console_level, file_level

    def __select_model(self, model_path, auto):
        models_dir = Path(model_path) 
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            self.logger.error("사용 가능한 체크포인트를 찾을 수 없습니다.")
            exit(1)
        
        # 수정 시간 기준으로 정렬된 체크포인트 리스트 생성
        sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)
        current_index = 0
        
        if auto:
            current_model = sorted_models[0]
            current_model_learning_info = self.__find_learning_info_file(model_path, current_model)
            return current_model, current_model_learning_info
        else:
            while True:
                current_model = sorted_models[current_index]
                print(f"\n현재 선택된 모델: {current_model}")
                user_input = input("이 모델로 학습을 진행하시겠습니까? (y/n): ")
                
                if user_input.lower() == 'y':
                    current_model_learning_info = self.__find_learning_info_file(model_path, current_model)
                    print(f"학습 정보 파일: {current_model_learning_info}")
                    return current_model, current_model_learning_info

                elif user_input.lower() == 'n':
                    current_index = (current_index + 1) % len(sorted_models)
                    if current_index == 0:
                        print("\n모든 모델을 확인했습니다. 처음부터 다시 시작합니다.")
                else:
                    print("학습을 취소합니다.")
                    exit(1)

    def __find_learning_info_file(self, model_path, current_model):
        """
        learning_info 폴더에서 해당 모델의 학습 정보 파일을 찾습니다.
        """
        # 모델 파일 경로에서 파일 이름 추출
        if isinstance(current_model, Path):
            model_name = current_model.stem  # 확장자 제외한 파일 이름
        else:
            model_name = os.path.splitext(os.path.basename(current_model))[0]
        # learning_info 폴더 경로 구성
        learning_info_dir = f'{model_path}/learning_info'
        if not os.path.exists(learning_info_dir):
            return None
            
        # 동일한 이름의 JSON 파일 찾기
        json_file = os.path.join(learning_info_dir, f"{model_name}.json")
        if os.path.exists(json_file):
            return json_file
            
        return None
    
    def _load_model(self, further, auto):
        model_path, learning_info_path = self.__select_model('models' if further else 'checkpoints', auto)
        if model_path is None:
            return None, None
        
        model_info = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        with open(learning_info_path, 'r') as f:
            learning_info = json.load(f)

        # 모델 기본 정보
        model_name = model_info.get('model_name', 'ppo')
        state_dim = model_info.get('state_dim', self.env.observation_space.shape[0])
        action_dim = model_info.get('action_dim', self.env.action_space.n)
        
        # 학습 파라미터 (새 구조)
        learning_params = learning_info.get('learning_params', {})
        gamma = learning_params.get('gamma', 0.99)
        epsilon = learning_params.get('epsilon', 0.2)
        batch_size = learning_params.get('batch_size', 32)
        epochs = learning_params.get('epochs', 20)
        
        # 학습 진행 상태 (새 구조)
        training_state = learning_info.get('training_state', {})
        current_episode = training_state.get('current_episode', 0)
        self.checkpoint_term = training_state.get('checkpoint_term', 100)
        self.num_episodes = training_state.get('total_episodes', 2000)
        remaining_episodes = self.num_episodes - current_episode
        
        # 옵티마이저에서 학습률 가져오기
        optimizer_state = model_info['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][-1]['lr']  # critic의 학습률
        
        # 학습 결과
        training_results = learning_info.get('training_results', {})
        rewards_history = training_results.get('rewards_history', [])
        episode_results = training_results.get('episode_results', [])
        total_episodes = training_results.get('total_episodes', 0)
        completed_episodes = training_results.get('completed_episodes', 0)
        win_rate = training_results.get('win_rate', 0.0)
        episode_win_rate = training_results.get('episode_win_rate', [])
        profit_rate_history = training_results.get('profit_rate_history', [])
        all_balance_history = training_results.get('all_balance_history', [])
        step_num_history = training_results.get('step_num_history', [])

        if further:
            if auto:
                self.num_episodes = current_episode + 1000
            else:
                self.num_episodes = current_episode + int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 1000): ") or "1000")
            
                print(f"이전 학습률 설정:")
                print(f"- Actor 학습률: {lr_actor}")
                print(f"- Critic 학습률: {lr_critic}")
                # 학습 파라미터 조정 여부 확인
                adjust = input("학습률을 조정하시겠습니까? (y/n): ").lower() == 'y'
                if adjust:
                    try:
                        new_lr_actor = float(input("새로운 Actor 학습률 (현재: 3e-4): ") or "3e-4")
                        new_lr_critic = float(input("새로운 Critic 학습률 (현재: 1e-3): ") or "1e-3")
                        
                        # 옵티마이저 학습률 조정
                        for param_group in optimizer_state['param_groups']:
                            if 'critic' in str(param_group['params']):
                                param_group['lr'] = new_lr_critic
                            else:
                                param_group['lr'] = new_lr_actor
                        
                        print(f"학습률이 조정되었습니다: Actor={new_lr_actor}, Critic={new_lr_critic}")
                    except ValueError:
                        print("올바른 숫자를 입력하지 않아 기본 학습률을 유지합니다.")        

        else:
            print(f"이전 학습 진행 상황: {current_episode}/{self.num_episodes} 에피소드")
            print(f"남은 에피소드 수: {remaining_episodes}")
            # 사용자에게 추가 에피소드 수 확인
            user_input = input(f"기존 목표({self.num_episodes})까지 계속 진행하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                try:
                    new_total = int(input("새로운 총 에피소드 수를 입력하세요: "))
                    if new_total > current_episode:
                        self.num_episodes = new_total
                    else:
                        print("현재 에피소드보다 큰 수를 입력해야 합니다.")
                        return
                except ValueError:
                    print("올바른 숫자를 입력하세요.")
                    return
                
        # PPO 에이전트 재생성
        ppo_agent = algo.PPO3(
            state_dim=state_dim,
            action_dim=action_dim,
            model_name=model_name,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon=epsilon,
            epochs=epochs
        )

        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(model_info['actor_critic_state_dict'])
        
        # 옵티마이저 상태 로드 (새로운 학습률 적용)
        optimizer_state = model_info['optimizer_state_dict']
        optimizer_state['param_groups'][0]['lr'] = lr_actor  # actor 학습률
        optimizer_state['param_groups'][-1]['lr'] = lr_critic  # critic 학습률
        ppo_agent.optimizer.load_state_dict(optimizer_state)


        # 모델 정보 구성
        model_info = {
            'model_name': model_info.get('model_name', 'ppo'),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'learning_params': {
                'gamma': gamma,
                'epsilon': epsilon,
                'epochs': epochs,
                'lr_actor': lr_actor,
                'lr_critic': lr_critic,
                'batch_size': batch_size,
                'device': str(ppo_agent.device)
            }
        }

        learning_info = { 
            # 학습 진행 상태
            'training_state': {
                'current_episode': current_episode,
                'total_episodes': self.num_episodes,
                'last_step': 0,
                'checkpoint_term': self.checkpoint_term
            },
            # 학습 결과
            'training_results': {
                'rewards_history': rewards_history,
                'episode_results': episode_results,
                'total_episodes': total_episodes,
                'completed_episodes': completed_episodes,
                'win_rate': win_rate,
                'episode_win_rate': episode_win_rate,
                'profit_rate_history': profit_rate_history,
                'all_balance_history': all_balance_history,
                'step_num_history': step_num_history
            },
            # 환경 정보
            'environment_info': {
                'data_path': self.env.path if hasattr(self.env, 'path') else None,
                'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                'training_period': {
                    'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None,
                    'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None
                }
            },
            # 세션 정보
            'session_info': {
                'session_type': 'new',
                'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'start_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'log_file': f'logs/{model_name}.log',
                'previous_episodes': 0,
                'current_session_episodes': 0,
                'total_episodes_all_sessions': 0,
            }
        }

        return ppo_agent, model_info, learning_info
        
    def _set_model(self):
        model_name = input('\n모델 저장 이름을 입력해주세요. [default: ppo]: ') or "ppo"
        self.num_episodes = int(input('학습할 에피소드 수를 입력해주세요. [default: 1000]: ') or "1000")
        self.checkpoint_term = int(input('체크포인트 저장 주기를 입력해주세요. [default: 100]: ') or "100")
        lr_actor = float(input('lr_actor [default: 3e-4]: ') or "3e-4")            
        lr_critic = float(input('lr_critic [default: 1e-3]: ') or "1e-3")
        gamma = float(input('gamma [default: 0.99]: ') or "0.99")
        epsilon = float(input('epsilon [default: 0.2]: ') or "0.2")
        batch_size = int(input('batch_size [default: 32]: ') or "32")
        epochs = int(input('epochs [default: 20]: ') or "20")

        ppo_agent = algo.PPO3(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                model_name=model_name,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                epsilon=epsilon,
                epochs=epochs)

        # 새 모델의 초기 정보 구성 - 새로운 데이터 구조 적용
        model_info = {
            # 모델 기본 정보
            'model_name': model_name,
            'state_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.n,
            
            # 모델 가중치는 초기 상태로 유지
            
            # 학습 파라미터
            'learning_params': {
                'gamma': gamma,
                'epsilon': epsilon,
                'epochs': epochs,
                'lr_actor': lr_actor,
                'lr_critic': lr_critic,
                'batch_size': batch_size,
                'device': str(ppo_agent.device)
            },
        }

        learning_info = { 
            # 학습 진행 상태
            'training_state': {
                'current_episode': 0,
                'total_episodes': self.num_episodes,
                'last_step': 0,
                'checkpoint_term': self.checkpoint_term
            },
            
            # 학습 결과 - 초기 상태
            'training_results': {
                'rewards_history': [],
                'episode_results': [],
                'total_episodes': 0,
                'completed_episodes': 0,
                'win_rate': 0.0,
                'episode_win_rate': [],
                'profit_rate_history': [],
                'all_balance_history': [],
                'step_num_history': []
            },
            
            # 환경 정보
            'environment_info': {
                'data_path': self.env.path if hasattr(self.env, 'path') else None,
                'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                'training_period': {
                    'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None,
                    'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None
                }
            },
            
            # 세션 정보 - 초기 상태
            'session_info': {
                'session_type': 'new',
                'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'start_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'log_file': f'logs/{model_name}.log',
                'previous_episodes': 0,
                'current_session_episodes': 0,
                'total_episodes_all_sessions': 0,
                'previous_steps': 0,
                'current_session_steps': 0,
                'training_sessions': 1
            }
        }

        return ppo_agent, model_info, learning_info

    def train(self, **kwargs):

        self.logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))

        # 학습 진행 상황
        if 'training_state' in self.learning_info:
            training_state = self.learning_info['training_state']
            start_episode = training_state.get('current_episode', 0)
            end_episode = training_state.get('total_episodes', self.num_episodes)
            checkpoint_term = training_state.get('checkpoint_term', 100)
        
        # 성능 지표
        if 'training_results' in self.learning_info:
            training_results = self.learning_info['training_results']
            episode_rewards = training_results.get('rewards_history', [])
            episode_results = training_results.get('episode_results', [])
            win_rate = training_results.get('win_rate', 0.0)
            episode_win_rate = training_results.get('episode_win_rate', [])
            profit_rate_history = training_results.get('profit_rate_history', [])
            all_balance_history = training_results.get('all_balance_history', [])
            step_num_history = training_results.get('step_num_history', [])

        # 이전 학습 시간 로드 (없으면 현재 시간 사용)
        start_time = self.learning_info.get('start_time', (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        
        is_normal_exit = False
        episode = start_episode  # try 블록 밖에서 초기화
        
        try:
            for episode in range(start_episode, end_episode):
                self.logger.render_episode_start(episode + 1)
                balance_history = []
                actions = []
                state = self.env.reset()

                episode_reward = 0                
                update_count = 0
                
                while True:
                    self.env.num += 1
                    action, value, log_prob = self.agent.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    
                    actions.append(info['position'])
                    
                    self.agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                    
                    episode_reward += reward
                    state = next_state
                    
                    if len(self.agent.memory) >= self.agent.batch_size:
                        #update_count += agent.update(success_rate=sum(episode_results) / len(episode_results))
                        update_count += self.agent.update()
                    balance_history.append(info['balance'])
                    if done:
                        profit_rate_history.append(info['profit_rate'])
                        episode_results.append(1 if self.env.balance > self.env.initial_balance * 1.01 else 0)
                        update_count += self.agent.update(success_rate=sum(episode_results) / len(episode_results)) if info['liquidated'] else 0
                        break

                self.logger.render(f"  반복한 step: {self.env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
                self.env.render()
                self.logger.render_episode_end(sum(episode_results) / len(episode_results))
                
                episode_rewards.append(episode_reward)
                all_balance_history.append(self.env.balance)
                step_num_history.append(self.env.num)
                
                # 에피소드 내에서 승률 계산
                profit_count = sum(1 for p in self.env.profit_history if p > 0)
                loss_count = sum(1 for p in self.env.profit_history if p < 0)
                total_trades = profit_count + loss_count
                win_rate = profit_count / total_trades if total_trades > 0 else 0
                episode_win_rate.append(win_rate)

                # 학습 진행 상황 평가 및 시각화 (50 에피소드마다)
                if (episode + 1) % 50 == 0:
                    # 학습 진행 상황 평가 및 시각화
                    os.makedirs(f'results/{self.agent.model_name}/learning', exist_ok=True)
                    os.makedirs(f'results/{self.agent.model_name}/performance', exist_ok=True)
                    metrics = plot_episode_metrics(
                        balance_history=self.env.balance_history,
                        profit_history=self.env.profit_history,
                        profit_rate_history=self.env.profit_rate_history,
                        price_history=self.env.price_history,
                        actions=actions,
                        balance_profit_rate_history=self.env.balance_profit_rate_history,
                        path=f'results/{self.agent.model_name}/learning/{self.agent.model_name}_learning_{episode + 1}.png'
                    )
                    self.agent.plot_performance(f'results/{self.agent.model_name}/performance/{self.agent.model_name}_performance_{episode + 1}.png')
                    # 학습 지표 로깅
                    self.logger.render(f"  학습 지표 - 승률: {metrics['win_rate']:.2%}")
                if (episode + 1) % 500 == 0:
                    metrics = plot_learning_progress(
                        episode_win_rate=episode_win_rate,
                        profit_rate_history=profit_rate_history,
                        episode_rewards=episode_rewards,
                        episode_results=episode_results,
                        path=f'results/{self.agent.model_name}/{self.agent.model_name}_result_{episode + 1}.png'
                    )
                # 주기적으로 체크포인트 저장
                if (episode + 1) % checkpoint_term == 0:

                    learning_info = {
                        # 학습 진행 상태
                        'training_state': {
                            'current_episode': episode + 1,
                            'total_episodes': self.num_episodes,
                            'last_step': self.env.last_step,
                            'checkpoint_term': checkpoint_term
                        },
                        
                        # 학습 결과
                        'training_results': {
                            'rewards_history': episode_rewards,
                            'episode_results': episode_results,
                            'completed_episodes': sum(episode_results),
                            'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0,
                            'episode_win_rate': episode_win_rate,
                            'profit_rate_history': profit_rate_history if 'profit_rate_history' in locals() else [],
                            'all_balance_history': all_balance_history if 'all_balance_history' in locals() else [],
                            'step_num_history': step_num_history if 'step_num_history' in locals() else []
                        },
                        
                        # 환경 정보
                        'environment_info': {
                            'data_path': self.env.path if hasattr(self.env, 'path') else None,
                            'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                            'training_period': {
                                'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                                'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                            }
                        },
                        
                        # 세션 정보
                        'session_info': {
                            'session_type': 'checkpoint',
                            'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                            'start_time': start_time,
                            'log_file': f'logs/{self.agent.model_name}.log',
                            'previous_checkpoints': self.model_info.get('previous_checkpoints', []),
                            'previous_episodes': self.model_info.get('start_episode', 0),
                            'current_session_episodes': episode + 1 - self.model_info.get('start_episode', 0),
                            'training_sessions': self.model_info.get('training_sessions', 0) + 1
                        }
                    }
                    time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")
                    os.makedirs('checkpoints', exist_ok=True)
                    os.makedirs(f'checkpoints/learning_info', exist_ok=True)
                    self.agent.save_model(f'checkpoints/{self.agent.model_name}_{time}_ep_{episode + 1}.pth')
                    self.agent.save_learning_state(learning_info, f'checkpoints/learning_info/{self.agent.model_name}_{time}_ep_{episode + 1}.json')
                    
                    self.logger.render(f" <체크포인트 저장됨: {episode + 1}>")
            
            is_normal_exit = True

            result = {
                'total_episodes': self.num_episodes,
                'completed_episodes': sum(episode_results),
                'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0
            }

            self.logger.render_training_result(result=result)

        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:
                if is_normal_exit:
                    self.logger.render_training_end(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
                else:
                    self.logger.render_training_stop(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
                
                time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")
                if is_normal_exit:
                    episode += 1
                # 체크포인트 데이터 준비
                learning_info = {
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode,
                        'total_episodes': self.num_episodes,
                        'last_step': self.env.last_step,
                        'checkpoint_term': checkpoint_term
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'rewards_history': episode_rewards[:episode],
                        'episode_results': episode_results[:episode],
                        'completed_episodes': sum(episode_results[:episode]),
                        'win_rate': sum(episode_results[:episode]) / len(episode_results[:episode]) * 100 if episode_results else 0,
                        'episode_win_rate': episode_win_rate[:episode],
                        'profit_rate_history': profit_rate_history[:episode],
                        'all_balance_history': all_balance_history[:episode],
                        'step_num_history': step_num_history[:episode]
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.env.path if hasattr(self.env, 'path') else None,
                        'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                        'training_period': {
                            'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                            'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'checkpoint',
                        'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_checkpoints': self.model_info.get('previous_checkpoints', []),
                        'previous_episodes': self.model_info.get('start_episode', 0),
                        'current_session_episodes': episode + 1 - self.model_info.get('start_episode', 0),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    }
                }
                    
                # 수익률 통계 계산
                balance_profit_rate_array = np.array(all_balance_history) if 'all_balance_history' in locals() and all_balance_history else np.array([])
                balance_profit_rate_std = float(np.std(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0
                
                # 승률 계산
                win_rate = sum(episode_results) / len(episode_results) * 100 if episode_results else 0
                
                # 수익률 데이터
                profit_rates = np.array(profit_rate_history) if 'profit_rate_history' in locals() and profit_rate_history else np.array([])
                
                metadata = {
                    # 모델 기본 정보
                    'model_name': self.agent.model_name,
                    'training_start_time': start_time,  # 이번 학습 시작 시간
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': self.agent.state_dim,
                        'action_dim': self.agent.action_dim,
                        'gamma': self.agent.gamma,
                        'epsilon': self.agent.epsilon,
                        'epochs': self.agent.epochs,
                        'lr_actor': self.agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': self.agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': self.agent.batch_size,  # 메모리 크기
                        'device': str(self.agent.device)
                    },
                    
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode + 1,
                        'total_episodes': self.num_episodes,
                        'last_step': self.env.last_step,
                        'checkpoint_term': checkpoint_term,
                        'total_steps': self.env.num if hasattr(self.env, 'num') else 0,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'completed_episodes': sum(episode_results) if episode_results else 0,
                        'win_rate': win_rate,
                        'episode_win_rate': episode_win_rate,
                        'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                        'max_episode_win_rate': max(episode_win_rate) if len(episode_win_rate) > 0 else 0,
                        'min_episode_win_rate': min(episode_win_rate) if len(episode_win_rate) > 0 else 0
                    },
                    
                    # 수익률 통계
                    'returns_stats': {
                        'best_return': float(max(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'worst_return': float(min(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'final_return': float(balance_profit_rate_array[-1]) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'mean_return': float(np.mean(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0,
                        'return_std': balance_profit_rate_std,
                        'sharpe_ratio': float(np.mean(balance_profit_rate_array) / balance_profit_rate_std) if balance_profit_rate_std != 0 and len(balance_profit_rate_array) > 0 else 0
                    },
                    
                    # 수익률 데이터 통계
                    'profit_rate_stats': {
                        'best_profit_rate': float(max(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'worst_profit_rate': float(min(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'final_profit_rate': float(profit_rates[-1]) if len(profit_rates) > 0 else float('-inf'),
                        'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0,
                        'profit_rate_std': float(np.std(profit_rates)) if len(profit_rates) > 0 else 0,
                        'positive_rate': float(np.sum(profit_rates > 0) / len(profit_rates)) if len(profit_rates) > 0 else 0
                    },
                    
                    # 보상 통계
                    'reward_stats': {
                        'total_reward': sum(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'mean_reward': np.mean(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'max_reward': max(episode_rewards) if 'episode_rewards' in locals() else float('-inf'),
                        'min_reward': min(episode_rewards) if 'episode_rewards' in locals() else float('inf'),
                        'reward_std': float(np.std(episode_rewards)) if 'episode_rewards' in locals() else 0
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.env.path if hasattr(self.env, 'path') else None,
                        'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                        'training_period': {
                            'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                            'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'completed' if is_normal_exit else 'interrupted',
                        'session_time': time,
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_episodes': start_episode,
                        'current_session_episodes': episode + 1 - start_episode,
                        'total_episodes_all_sessions': start_episode + (episode + 1 - start_episode),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    },
                    
                    # 학습 히스토리
                    'training_history': {
                        'previous_sessions': self.model_info.get('training_history', {}).get('previous_sessions', []),
                        'current_session': {
                            'session_number': self.model_info.get('training_sessions', 0) + 1,
                            'start_episode': start_episode,
                            'end_episode': episode + 1,
                            'episodes_trained': episode + 1 - start_episode,
                            'start_time': start_time,
                            'end_time': time,
                            'win_rate': win_rate,
                            'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                            'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0
                        }
                    }
                }
                
                if is_normal_exit:
                    os.makedirs('models', exist_ok=True)
                    os.makedirs(f'models/learning_info', exist_ok=True)
                    os.makedirs(f'json/{self.agent.model_name}', exist_ok=True)
                    os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)
                    
                    self.agent.save_model(f'models/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'models/learning_info/{self.agent.model_name}_{time}.json')
                    with open(f'json/{self.agent.model_name}/{self.agent.model_name}_metadata_{time}.json', 'w') as f:
                        json.dump(metadata, f, indent=4)
                    
                    # 최종 학습 진행 상황 평가 및 시각화
                    metrics = plot_learning_progress(
                        episode_win_rate=episode_win_rate,
                        profit_rate_history=profit_rate_history,
                        episode_rewards=episode_rewards,
                        episode_results=episode_results,
                        path=f'results/{self.agent.model_name}/{self.agent.model_name}_result_{time}.png'
                    )

                else:
                    os.makedirs('checkpoints', exist_ok=True)
                    os.makedirs(f'models/learning_info', exist_ok=True)
                    os.makedirs('json', exist_ok=True)
                    
                    # 체크포인트 데이터에 중단 상태 표시
                    learning_info['session_info']['session_type'] = 'interrupted'
                    
                    self.agent.save_model(f'checkpoints/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'checkpoints/learning_info/{self.agent.model_name}_{time}.json')
                    with open(f'json/{self.agent.model_name}/{self.agent.model_name}_checkpoint_{time}.json', 'w') as f:
                        json.dump(metadata, f, indent=4)

                self.logger.render(f" <체크포인트가 저장되었습니다: {time}>")
                
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return episode_rewards, {
            'total_steps': self.env.num,
            'start_time': start_time,
            'episode_results': episode_results,
            'learning_metrics': metrics if 'metrics' in locals() else None
        }
    
    def train_in_server(self):
        

        # 학습 진행 상황
        if 'training_state' in self.learning_info:
            training_state = self.learning_info['training_state']
            start_episode = training_state.get('current_episode', 0)
            end_episode = training_state.get('total_episodes', self.num_episodes)
            checkpoint_term = training_state.get('checkpoint_term', 100)
        # 성능 지표
        if 'training_results' in self.learning_info:
            training_results = self.learning_info['training_results']
            episode_rewards = training_results.get('rewards_history', [])
            episode_results = training_results.get('episode_results', [])
            win_rate = training_results.get('win_rate', 0.0)
            episode_win_rate = training_results.get('episode_win_rate', [])
            profit_rate_history = training_results.get('profit_rate_history', [])
            all_balance_history = training_results.get('all_balance_history', [])
            step_num_history = training_results.get('step_num_history', [])

        # 이전 학습 시간 로드 (없으면 현재 시간 사용)
        start_time = self.learning_info.get('start_time', (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        
        is_normal_exit = False
        episode = start_episode  # try 블록 밖에서 초기화
        
        try:
            for episode in range(start_episode, end_episode):
                balance_history = []
                actions = []
                state = self.env.reset()

                episode_reward = 0                
                update_count = 0
                
                while True:
                    self.env.num += 1
                    action, value, log_prob = self.agent.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    
                    actions.append(info['position'])
                    
                    self.agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                    
                    episode_reward += reward
                    state = next_state
                    
                    if len(self.agent.memory) >= 128:
                        update_count += self.agent.update()
                    balance_history.append(info['balance'])
                    if done:
                        profit_rate_history.append(info['profit_rate'])
                        episode_results.append(1 if self.env.balance > self.env.initial_balance * 1.01 else 0)
                        update_count += self.agent.update(success_rate=sum(episode_results) / len(episode_results)) if info['liquidated'] else 0
                        break

                episode_rewards.append(episode_reward)
                all_balance_history.append(self.env.balance)
                step_num_history.append(self.env.num)
                episode_win_rate.append(sum(1 if p > 0 else 0 for p in self.env.profit_history) / len(self.env.profit_history))

            
            is_normal_exit = True

            result = {
                'total_episodes': self.num_episodes,
                'completed_episodes': sum(episode_results),
                'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0
            }

        except KeyboardInterrupt:
            pass
        except Exception as e:
            pass
        finally:
            try:
             
                time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")
                if is_normal_exit:
                    episode += 1
                # 체크포인트 데이터 준비
                learning_info = {
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode,
                        'total_episodes': self.num_episodes,
                        'last_step': self.env.last_step,
                        'checkpoint_term': checkpoint_term
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'rewards_history': episode_rewards[:episode],
                        'episode_results': episode_results[:episode],
                        'completed_episodes': sum(episode_results[:episode]),
                        'win_rate': sum(episode_results[:episode]) / len(episode_results[:episode]) * 100 if episode_results else 0,
                        'episode_win_rate': episode_win_rate[:episode],
                        'profit_rate_history': profit_rate_history[:episode],
                        'all_balance_history': all_balance_history[:episode],
                        'step_num_history': step_num_history[:episode]
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.env.path if hasattr(self.env, 'path') else None,
                        'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                        'training_period': {
                            'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                            'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'checkpoint',
                        'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_checkpoints': self.model_info.get('previous_checkpoints', []),
                        'previous_episodes': self.model_info.get('start_episode', 0),
                        'current_session_episodes': episode + 1 - self.model_info.get('start_episode', 0),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    }
                }
                    
                # 수익률 통계 계산
                balance_profit_rate_array = np.array(all_balance_history) if 'all_balance_history' in locals() and all_balance_history else np.array([])
                balance_profit_rate_std = float(np.std(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0
                
                # 승률 계산
                win_rate = sum(episode_results) / len(episode_results) * 100 if episode_results else 0
                
                # 수익률 데이터
                profit_rates = np.array(profit_rate_history) if 'profit_rate_history' in locals() and profit_rate_history else np.array([])
                
                metadata = {
                    # 모델 기본 정보
                    'model_name': self.agent.model_name,
                    'training_start_time': start_time,  # 이번 학습 시작 시간
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': self.agent.state_dim,
                        'action_dim': self.agent.action_dim,
                        'gamma': self.agent.gamma,
                        'epsilon': self.agent.epsilon,
                        'epochs': self.agent.epochs,
                        'lr_actor': self.agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': self.agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': self.agent.batch_size,  # 메모리 크기
                        'device': str(self.agent.device)
                    },
                    
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode + 1,
                        'total_episodes': self.num_episodes,
                        'last_step': self.env.last_step,
                        'checkpoint_term': checkpoint_term,
                        'total_steps': self.env.num if hasattr(self.env, 'num') else 0,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'completed_episodes': sum(episode_results) if episode_results else 0,
                        'win_rate': win_rate,
                        'episode_win_rate': episode_win_rate,
                        'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                        'max_episode_win_rate': max(episode_win_rate) if len(episode_win_rate) > 0 else 0,
                        'min_episode_win_rate': min(episode_win_rate) if len(episode_win_rate) > 0 else 0
                    },
                    
                    # 수익률 통계
                    'returns_stats': {
                        'best_return': float(max(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'worst_return': float(min(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'final_return': float(balance_profit_rate_array[-1]) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'mean_return': float(np.mean(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0,
                        'return_std': balance_profit_rate_std,
                        'sharpe_ratio': float(np.mean(balance_profit_rate_array) / balance_profit_rate_std) if balance_profit_rate_std != 0 and len(balance_profit_rate_array) > 0 else 0
                    },
                    
                    # 수익률 데이터 통계
                    'profit_rate_stats': {
                        'best_profit_rate': float(max(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'worst_profit_rate': float(min(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'final_profit_rate': float(profit_rates[-1]) if len(profit_rates) > 0 else float('-inf'),
                        'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0,
                        'profit_rate_std': float(np.std(profit_rates)) if len(profit_rates) > 0 else 0,
                        'positive_rate': float(np.sum(profit_rates > 0) / len(profit_rates)) if len(profit_rates) > 0 else 0
                    },
                    
                    # 보상 통계
                    'reward_stats': {
                        'total_reward': sum(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'mean_reward': np.mean(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'max_reward': max(episode_rewards) if 'episode_rewards' in locals() else float('-inf'),
                        'min_reward': min(episode_rewards) if 'episode_rewards' in locals() else float('inf'),
                        'reward_std': float(np.std(episode_rewards)) if 'episode_rewards' in locals() else 0
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.env.path if hasattr(self.env, 'path') else None,
                        'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                        'training_period': {
                            'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                            'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'completed' if is_normal_exit else 'interrupted',
                        'session_time': time,
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_episodes': start_episode,
                        'current_session_episodes': episode + 1 - start_episode,
                        'total_episodes_all_sessions': start_episode + (episode + 1 - start_episode),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    },
                    
                    # 학습 히스토리
                    'training_history': {
                        'previous_sessions': self.model_info.get('training_history', {}).get('previous_sessions', []),
                        'current_session': {
                            'session_number': self.model_info.get('training_sessions', 0) + 1,
                            'start_episode': start_episode,
                            'end_episode': episode + 1,
                            'episodes_trained': episode + 1 - start_episode,
                            'start_time': start_time,
                            'end_time': time,
                            'win_rate': win_rate,
                            'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                            'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0
                        }
                    }
                }
                

                if is_normal_exit:
                    os.makedirs('saved_models', exist_ok=True)
                    os.makedirs('saved_models/learning_info', exist_ok=True)
                    os.makedirs('json', exist_ok=True)
                    os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)

                    # model은 models 폴더에 저장, 쉘 스크립트로 AI_Lambda 레포지토리에 업로드
                    # learning_info는 저장할 필요 없음
                    # metadata는 AWS S3에 저장, 
                    

                    self.agent.save_model(f'saved_models/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'saved_models/learning_info/{self.agent.model_name}_{time}.json')                    
                    with open(f'json/{self.agent.model_name}_metadata_{time}.json', 'w') as f:
                        json.dump(metadata, f, indent=4)
                    
               

                else:
                    os.makedirs('checkpoints', exist_ok=True)
                    os.makedirs('checkpoints/learning_info', exist_ok=True)
                    os.makedirs('json', exist_ok=True)
                    
                    # 체크포인트 데이터에 중단 상태 표시
                    learning_info['session_info']['session_type'] = 'interrupted'
                    
                    # 이전 모델 파일들 삭제
                    pth_files = glob.glob('checkpoints/*.pth')
                    for file in pth_files:
                        try:
                            os.remove(file)
                        except Exception as e:
                            pass
                    
                    # models/learning_info 폴더의 .json 파일 삭제
                    json_files = glob.glob('checkpoints/learning_info/*.json')
                    for file in json_files:
                        try:
                            os.remove(file)
                        except Exception as e:
                            pass
                    
                    # metadata 폴더의 .json 파일 삭제
                    metadata_files = glob.glob('metadata/*.json')
                    for file in metadata_files:
                        try:
                            os.remove(file)
                        except Exception as e:
                            pass

                    self.agent.save_model(f'checkpoints/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'checkpoints/learning_info/{self.agent.model_name}_{time}.json')
                    with open(f'json/{self.agent.model_name}_checkpoint_{time}.json', 'w') as f:
                        json.dump(metadata, f, indent=4)

            except Exception as save_error:
                pass
        
        return episode_rewards, {
            'total_steps': self.env.num,
            'start_time': start_time,
            'episode_results': episode_results
        }
    

    def test(self):
        self.logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        
        # 학습 진행 상황
        start_episode = 0
        end_episode = 20
        # 성능 지표
        episode_rewards = []
        episode_results = []
        win_rate = 0.0
        episode_win_rate = []
        profit_rate_history = []
        all_balance_history = []
        step_num_history = []

        # 이전 학습 시간 로드 (없으면 현재 시간 사용)
        start_time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S')
        
        try:
            for episode in range(end_episode):
                self.logger.render_episode_start(episode + 1)
                balance_history = []
                actions = []
                state = self.test_env.reset()

                episode_reward = 0                
                update_count = 0
                
                while True:
                    self.test_env.num += 1
                    action, value, log_prob = self.agent.select_action(state)
                    next_state, reward, done, info = self.test_env.step(action)
                    
                    actions.append(info['position'])
                    
                    episode_reward += reward
                    state = next_state
                    balance_history.append(info['balance'])

                    if done:
                        profit_rate_history.append(info['profit_rate'])
                        episode_results.append(1 if self.test_env.balance > self.test_env.initial_balance * 1.01 else 0)
                        break

                self.logger.render(f"  반복한 step: {self.test_env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
                self.test_env.render()
                self.logger.render_episode_end(sum(episode_results) / len(episode_results))
                
                episode_rewards.append(episode_reward)
                all_balance_history.append(self.test_env.balance)
                step_num_history.append(self.test_env.num)
                
                # 에피소드 내에서 승률 계산
                profit_count = sum(1 for p in self.test_env.profit_history if p > 0)
                loss_count = sum(1 for p in self.test_env.profit_history if p < 0)
                total_trades = profit_count + loss_count
                win_rate = profit_count / total_trades if total_trades > 0 else 0
                episode_win_rate.append(win_rate)

                os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)
                os.makedirs(f'results/{self.agent.model_name}/test', exist_ok=True)
                metrics = plot_episode_metrics(
                        balance_history=self.test_env.balance_history,
                        profit_history=self.test_env.profit_history,
                        profit_rate_history=self.test_env.profit_rate_history,
                        price_history=self.test_env.price_history,
                        actions=actions,
                        balance_profit_rate_history=self.test_env.balance_profit_rate_history,
                        path=f'results/{self.agent.model_name}/test/{self.agent.model_name}_learning_{episode + 1}.png'
                    )

            result = {
                'total_episodes': self.num_episodes,
                'completed_episodes': sum(episode_results),
                'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0
            }

            self.logger.render_training_result(result=result)

        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:
                
                time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")
                    
                # 수익률 통계 계산
                balance_profit_rate_array = np.array(all_balance_history) if 'all_balance_history' in locals() and all_balance_history else np.array([])
                balance_profit_rate_std = float(np.std(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0
                
                # 승률 계산
                win_rate = sum(episode_results) / len(episode_results) * 100 if episode_results else 0
                
                # 수익률 데이터
                profit_rates = np.array(profit_rate_history) if 'profit_rate_history' in locals() and profit_rate_history else np.array([])
                
                metadata = {
                    # 모델 기본 정보
                    'model_name': self.agent.model_name,
                    'training_start_time': start_time,  # 이번 학습 시작 시간
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': self.agent.state_dim,
                        'action_dim': self.agent.action_dim,
                        'gamma': self.agent.gamma,
                        'epsilon': self.agent.epsilon,
                        'epochs': self.agent.epochs,
                        'lr_actor': self.agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': self.agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': self.agent.batch_size,  # 메모리 크기
                        'device': str(self.agent.device)
                    },
                    
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode + 1,
                        'total_episodes': self.num_episodes,
                        'last_step': self.test_env.last_step,
                        'total_steps': self.test_env.num if hasattr(self.test_env, 'num') else 0,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'completed_episodes': sum(episode_results) if episode_results else 0,
                        'win_rate': win_rate,
                        'episode_win_rate': episode_win_rate,
                        'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                        'max_episode_win_rate': max(episode_win_rate) if len(episode_win_rate) > 0 else 0,
                        'min_episode_win_rate': min(episode_win_rate) if len(episode_win_rate) > 0 else 0
                    },
                    
                    # 수익률 통계
                    'returns_stats': {
                        'best_return': float(max(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'worst_return': float(min(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'final_return': float(balance_profit_rate_array[-1]) if len(balance_profit_rate_array) > 0 else float('-inf'),
                        'mean_return': float(np.mean(balance_profit_rate_array)) if len(balance_profit_rate_array) > 0 else 0,
                        'return_std': balance_profit_rate_std,
                        'sharpe_ratio': float(np.mean(balance_profit_rate_array) / balance_profit_rate_std) if balance_profit_rate_std != 0 and len(balance_profit_rate_array) > 0 else 0
                    },
                    
                    # 수익률 데이터 통계
                    'profit_rate_stats': {
                        'best_profit_rate': float(max(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'worst_profit_rate': float(min(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'final_profit_rate': float(profit_rates[-1]) if len(profit_rates) > 0 else float('-inf'),
                        'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0,
                        'profit_rate_std': float(np.std(profit_rates)) if len(profit_rates) > 0 else 0,
                        'positive_rate': float(np.sum(profit_rates > 0) / len(profit_rates)) if len(profit_rates) > 0 else 0
                    },
                    
                    # 보상 통계
                    'reward_stats': {
                        'total_reward': sum(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'mean_reward': np.mean(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'max_reward': max(episode_rewards) if 'episode_rewards' in locals() else float('-inf'),
                        'min_reward': min(episode_rewards) if 'episode_rewards' in locals() else float('inf'),
                        'reward_std': float(np.std(episode_rewards)) if 'episode_rewards' in locals() else 0
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.test_env.path if hasattr(self.test_env, 'path') else None,
                        'total_data_length': len(self.test_env.data) if hasattr(self.test_env, 'data') else 0,
                        'training_period': {
                            'start': str(self.test_env.data.index[0]) if hasattr(self.test_env, 'data') else None,
                            'end': str(self.test_env.data.index[-1]) if hasattr(self.test_env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'test',
                        'session_time': time,
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_episodes': start_episode,
                        'current_session_episodes': episode + 1 - start_episode,
                        'total_episodes_all_sessions': start_episode + (episode + 1 - start_episode),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    },
                    
                    # 학습 히스토리
                    'training_history': {
                        'previous_sessions': self.model_info.get('training_history', {}).get('previous_sessions', []),
                        'current_session': {
                            'session_number': self.model_info.get('training_sessions', 0) + 1,
                            'start_episode': start_episode,
                            'end_episode': episode + 1,
                            'episodes_trained': episode + 1 - start_episode,
                            'start_time': start_time,
                            'end_time': time,
                            'win_rate': win_rate,
                            'mean_episode_win_rate': float(np.mean(episode_win_rate)) if len(episode_win_rate) > 0 else 0,
                            'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0
                        }
                    }
                }
                
                os.makedirs('models', exist_ok=True)
                os.makedirs(f'models/learning_info', exist_ok=True)
                os.makedirs('json', exist_ok=True)
                os.makedirs(f'json/{self.agent.model_name}', exist_ok=True)
                os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)
                
                with open(f'json/{self.agent.model_name}/{self.agent.model_name}_metadata_test_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # 최종 학습 진행 상황 평가 및 시각화
                metrics = plot_learning_progress(
                    episode_win_rate=episode_win_rate,
                    profit_rate_history=profit_rate_history,
                    episode_rewards=episode_rewards,
                    episode_results=episode_results,
                    path=f'results/{self.agent.model_name}/{self.agent.model_name}_test_{time}.png'
                )

                self.logger.render(f" <체크포인트가 저장되었습니다: {time}>")
                
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return episode_rewards, {
            'total_steps': self.test_env.num,
            'start_time': start_time,
            'episode_results': episode_results,
            'learning_metrics': metrics if 'metrics' in locals() else None
        }