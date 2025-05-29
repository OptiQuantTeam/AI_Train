import torch
from pathlib import Path
import algorithm as algo
import env
from Logger import Logger, LogLevel
import datetime
import numpy as np
import os
import json
import glob

class AutoLoader():
    def __init__(self, env_path):
        self.env = env.FuturesEnv(path=env_path)

        self.agent, self.model_info, self.learning_info = self._load_model()
        self.logger = Logger(self.agent.model_name, f'saved_model/logs/system.log', console_level=LogLevel.INFO, file_level=LogLevel.INFO)
        os.makedirs('saved_model/logs', exist_ok=True)
        self.env.logger = self.logger


    def __select_model(self, model_path):
        models_dir = Path(model_path) 
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            #self.logger.error("사용 가능한 체크포인트를 찾을 수 없습니다.")
            print("사용 가능한 체크포인트를 찾을 수 없습니다.")
            exit(1)
        
        # 수정 시간 기준으로 정렬된 체크포인트 리스트 생성
        sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)    
    
        current_model = sorted_models[0]
        current_model_learning_info = self.__find_learning_info_file(model_path, current_model)
        return current_model, current_model_learning_info

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
    
    def _load_model(self):
        model_path, learning_info_path = self.__select_model('saved_model')
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
        alpha = learning_params.get('alpha', 0.5)
        
        # 학습 진행 상태 (새 구조)
        training_state = learning_info.get('training_state', {})
        current_episode = training_state.get('current_episode', 0)
        self.checkpoint_term = training_state.get('checkpoint_term', 100)
        self.num_episodes = training_state.get('total_episodes', 2000)
        
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


        self.num_episodes = current_episode + 100
        # PPO 에이전트 재생성
        ppo_agent = algo.PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            model_name=model_name,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            batch_size=batch_size,
            epsilon=epsilon,
            epochs=epochs,
            alpha=alpha
        )
        print(model_name)
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
                'alpha': alpha,
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
        

    def _delete_old_models(self):
        """이전 모델 파일과 관련 정보를 삭제합니다."""
        try:
            if hasattr(self, 'agent'):
                # .pth 모델 파일 삭제
                pth_files = glob.glob('saved_model/*.pth')
                for file in pth_files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        self.logger.error(f"모델 파일 삭제 중 오류 발생: {str(e)}")
                
                # learning_info 폴더의 .json 파일 삭제
                json_files = glob.glob('saved_model/learning_info/*.json')
                for file in json_files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        self.logger.error(f"학습 정보 파일 삭제 중 오류 발생: {str(e)}")
                
                # metadata 폴더의 .json 파일 삭제
                metadata_files = glob.glob(f'saved_model/metadata/*.json')
                for file in metadata_files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        self.logger.error(f"메타데이터 파일 삭제 중 오류 발생: {str(e)}")

                self.logger.error("이전 모델 파일들과 로그 파일들이 성공적으로 삭제되었습니다.")
            else:
                self.logger.error("agent가 초기화되지 않아 파일 삭제를 진행할 수 없습니다.")
            
        except Exception as e:
            self.logger.error(f"파일 삭제 중 예상치 못한 오류 발생: {str(e)}")
    
    def train(self, **kwargs):
        self.logger.setTrainLevel()  # 로깅 레벨을 ERROR로 설정
        self.logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        self.agent.test_mode = False

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
                        update_count += self.agent.update()

                    balance_history.append(info['balance'])
                    if done:
                        profit_rate_history.append(info['profit_rate'])
                        episode_results.append(1 if self.env.balance > self.env.initial_balance else 0)
                        update_count += self.agent.update() if info['liquidated'] else 0
                        break

                self.logger.error(f"  반복한 step: {self.env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
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

            result = {
                'total_episodes': self.num_episodes,
                'completed_episodes': sum(episode_results),
                'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0
            }

            self.logger.render_training_result(result=result)
            self.agent.memory.clear()
        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:

                self.logger.render_training_end(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))

                time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H-%M-%S")

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
                        'rewards_history': episode_rewards,
                        'episode_results': episode_results,
                        'completed_episodes': sum(episode_results),
                        'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0,
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
                    

                os.makedirs('saved_model', exist_ok=True)
                os.makedirs(f'saved_model/learning_info', exist_ok=True)
                os.makedirs(f'saved_model/metadata', exist_ok=True)

                os.makedirs(f'output/', exist_ok=True)
                os.makedirs(f'output/learning_info', exist_ok=True)
                os.makedirs(f'output/metadata', exist_ok=True)
                
                self._delete_old_models()
                self.agent.save_model(f'saved_model/{self.agent.model_name}_{time}.pth')
                self.agent.save_model(f'output/{self.agent.model_name}_{time}.pth')
                self.agent.save_learning_state(learning_info, f'saved_model/learning_info/{self.agent.model_name}_{time}.json')
                self.agent.save_learning_state(learning_info, f'output/learning_info/{self.agent.model_name}_{time}.json')
                
                
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return episode_rewards

    def test(self):
        self.agent.test_mode = True
        self.logger.setTestLevel()

        try:
            self.logger.render_test_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))

            balance_history = []
            actions = []
            episode_reward = 0                
            
            state = self.env.reset()
            
            while True:
                self.env.num += 1
                action, value, log_prob = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                actions.append(info['position'])
                
                episode_reward += reward
                state = next_state


                balance_history.append(info['balance'])
                if done:
                    break

            self.env.render()
        
            result = {
                'environment_data': {
                    'episode_reward': episode_reward,
                    'final_balance': self.env.balance,
                    'initial_balance': self.env.initial_balance,
                    'total_steps': self.env.num,     
                },
                'performance_metrics': {
                    'total_profit': self.env.balance - self.env.initial_balance,
                    'profit_rate': (self.env.balance - self.env.initial_balance) / self.env.initial_balance * 100,
                    'profitable_trades': sum(1 for i in range(1, len(balance_history)) if balance_history[i] > balance_history[i-1]),
                    'average_profit_per_trade': (self.env.balance - self.env.initial_balance) / len(actions) if actions else 0
                },
                'trading_statistics': {
                    'long_positions': sum(1 for action in actions if action == 1),
                    'short_positions': sum(1 for action in actions if action == -1),
                    'neutral_positions': sum(1 for action in actions if action == 0),
                    'consecutive_wins': self._calculate_consecutive_wins(balance_history),
                    'consecutive_losses': self._calculate_consecutive_losses(balance_history),
                    'average_holding_time': self.env.num / len(actions) if actions else 0
                }
            }
            balance_profit_rate = self.env.balance_profit_rate * 100
            
            self.logger.render_test_result(result)
            time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S')
            self.logger.render_test_end(time)


        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:
                metadata = {
                    # 모델 기본 정보
                    'model_name': self.agent.model_name,
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
                        'alpha': self.agent.alpha,
                        'device': str(self.agent.device)
                    },
                    
                    # 성능 지표
                    'performance_metrics': {
                    'total_profit': self.env.balance - self.env.initial_balance,
                    'profit_rate': (self.env.balance - self.env.initial_balance) / self.env.initial_balance * 100,
                    'profitable_trades': sum(1 for i in range(1, len(balance_history)) if balance_history[i] > balance_history[i-1]),
                    'average_profit_per_trade': (self.env.balance - self.env.initial_balance) / len(actions) if actions else 0
                    },

                    # 테스트 거래 통계 
                    'trading_statistics': {
                        'long_positions': sum(1 for action in actions if action == 1),
                        'short_positions': sum(1 for action in actions if action == -1),
                        'neutral_positions': sum(1 for action in actions if action == 0),
                        'consecutive_wins': self._calculate_consecutive_wins(balance_history),
                        'consecutive_losses': self._calculate_consecutive_losses(balance_history),
                        'average_holding_time': self.env.num / len(actions) if actions else 0
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
                }
                '''
                with open(f'saved_model/metadata/{self.agent.model_name}_metadata_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)

                with open(f'output/metadata/{self.agent.model_name}_metadata_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                '''
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return balance_profit_rate

    def _calculate_consecutive_wins(self, balance_history):
        if len(balance_history) < 2:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(balance_history)):
            if balance_history[i] > balance_history[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _calculate_consecutive_losses(self, balance_history):
        if len(balance_history) < 2:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(balance_history)):
            if balance_history[i] < balance_history[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive