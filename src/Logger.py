import logging
from enum import Enum
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class Logger:
    def __init__(self, model_name, log_file_path, console_level=LogLevel.ERROR, file_level=LogLevel.INFO):
        self.model_name = model_name
        self.log_file_path = log_file_path
        
        # 로거 설정
        self.logger = logging.getLogger(f"{model_name}")
        self.logger.setLevel(min(console_level.value, file_level.value))
        
        # 파일 핸들러 설정
        self.file_handler = RotatingFileHandler(
            filename=self.log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=20,
            encoding='utf-8'
        )
        self.file_handler.setLevel(file_level.value)
        
        # 콘솔 핸들러 설정
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(console_level.value)
        
        # 포매터 설정
        hour = (int(datetime.now().strftime("%H")) + 9) % 24
        self.formatter = logging.Formatter('[%(asctime)s] :: %(message)s', 
                                         datefmt=f'%y%m%d-{hour:02d}:%M:%S')
        
        # 각 핸들러에 포매터 적용
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)
        
        # 로거에 핸들러 추가
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        
        # 로그 중복 방지
        self.logger.propagate = False

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def render_model_info(self, model_info):
        self.logger.error("########################################################")
        self.logger.error("========== 모델 정보 ==========")
        self.logger.error(f" 모델 이름: {model_info.get('model_name', 'N/A')}")
        self.logger.error(f" 상태 차원: {model_info.get('state_dim', 'N/A')}")
        self.logger.error(f" 행동 차원: {model_info.get('action_dim', 'N/A')}")
        
        # 학습 파라미터
        if 'learning_params' in model_info:
            learning_params = model_info['learning_params']
            self.logger.error("----- 학습 파라미터 -----")
            self.logger.error(f" Actor 학습률: {learning_params.get('lr_actor', 'N/A')}")
            self.logger.error(f" Critic 학습률: {learning_params.get('lr_critic', 'N/A')}")
            self.logger.error(f" 감마: {learning_params.get('gamma', 'N/A')}")
            self.logger.error(f" 입실론: {learning_params.get('epsilon', 'N/A')}")
            self.logger.error(f" 에포크: {learning_params.get('epochs', 'N/A')}")
        
        # 학습 진행 상황
        if 'training_state' in model_info:
            training_state = model_info['training_state']
            self.logger.error("----- 학습 진행 상황 -----")
            self.logger.error(f" 현재 에피소드: {training_state.get('current_episode', 0)}")
            self.logger.error(f" 총 에피소드: {training_state.get('total_episodes', 'N/A')}")
            self.logger.error(f" 체크포인트 주기: {training_state.get('checkpoint_term', 'N/A')}")
        
        # 성능 지표
        if 'training_results' in model_info:
            training_results = model_info['training_results']
            self.logger.error("----- 성능 지표 -----")
            self.logger.error(f" 총 스텝 수: {training_results.get('total_steps', 'N/A')}")
            self.logger.error(f" 완료된 에피소드: {training_results.get('completed_episodes', 'N/A')}")
            self.logger.error(f" 승률: {training_results.get('win_rate', 'N/A'):.2f}%")
        
        self.logger.error("########################################################")

    def render_episode_start(self, episode):
        self.logger.error('========================================================')
        self.logger.error(f'에피소드 {episode}')

    def render(self, message):
        self.logger.info(message)

    def render_episode_end(self, success_episodes_rate):
        self.logger.error(f'에피소드 성공률: {success_episodes_rate*100:.2f}%')
        self.logger.error('========================================================')

    def render_training_result(self, result):
        self.logger.error('========================================================')
        self.logger.error(f'----- 학습 결과 -----')
        self.logger.error(f' 총 에피소드: {result.get("total_episodes", "N/A")}')
        self.logger.error(f' 완료된 에피소드: {result.get("completed_episodes", "N/A")}')
        self.logger.error(f' 승률: {result.get("win_rate", "N/A"):.2f}%')
        self.logger.error('========================================================')
        self.logger.error(f' ')

    def render_training_start(self, time):
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.logger.error(f'               학습 시작 {time}')
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    def render_training_end(self, time):
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.logger.error(f'               학습 종료 {time}')
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    def render_training_stop(self, time):
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.logger.error(f'               학습 중단 {time}')
        self.logger.error('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    def render_step_state(self, state):
        self.logger.debug(state)
