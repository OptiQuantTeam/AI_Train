import torch
from AutoLoader import AutoLoader
import data
'''
AWS Train 서버에서 일주일마다 추가 학습 진행을 위한 프로그램
'''

if __name__ == "__main__":
    
    # 지난 일주일 데이터 로드
    env_path = data.getCurrentData("BTCUSDT", "30m", limit=336)
    loader = AutoLoader(env_path)
    loader.train()
    
