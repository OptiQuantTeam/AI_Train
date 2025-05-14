import pandas as pd
import numpy as np

def EMA(data, window):
    # EMA 계산
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    
    # EMA 기울기 계산
    ema_slope = ema.diff() / ema.shift(1)
    
    # 기울기 정규화
    normalized_slope = np.zeros(len(data))
    for i in range(window, len(data)):
        # window 크기만큼의 기울기 데이터 추출
        slope_window = ema_slope.iloc[i-window:i]
        # 현재 기울기 정규화
        mean = slope_window.mean()
        std = slope_window.std()
        normalized_slope[i] = (ema_slope.iloc[i] - mean) / (std + 1e-8)
    
    return pd.DataFrame({
        'EMA': ema,
        'EMA_Slope': ema_slope,
        'Normalized_Slope': normalized_slope
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # EMA 계산
    ema_result = EMA(data, window=10)
    print(ema_result)
