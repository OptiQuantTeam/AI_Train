from .RSI import RSI
import pandas as pd

def StochasticRSI(data, rsi_window=14, stoch_window=14):
    rsi = RSI(data, window=rsi_window)
    stoch_rsi = (rsi - rsi.rolling(window=stoch_window, min_periods=1).min()) / (
                rsi.rolling(window=stoch_window, min_periods=1).max() - rsi.rolling(window=stoch_window, min_periods=1).min())
    stoch_rsi = round(stoch_rsi*100, 2)
    
    return stoch_rsi

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # 스토캐스틱 RSI 계산
    stoch_rsi_result = StochasticRSI(data)
    print(stoch_rsi_result)
