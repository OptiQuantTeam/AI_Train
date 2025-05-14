import env.FuturesEnv11_train as FuturesEnv11_train
import algorithm as algo
from Logger import Logger, LogLevel

path='data/preprocess/BTCUSDT/BTCUSDT-15m-technical4.csv'
logger = Logger('SequencePPO', f'logs/SequencePPO.log', console_level=LogLevel.CRITICAL, file_level=LogLevel.CRITICAL)
# 환경과 에이전트 초기화
env = FuturesEnv11_train(path, logger)
agent = algo.SequencePPO(env, sequence_length=32, batch_size=8)

# 학습 루프
for episode in range(1000):
    # 시퀀스 수집
    sequences = agent.collect_sequences(num_sequences=10)
    
    # 학습
    agent.train(sequences)
    
    