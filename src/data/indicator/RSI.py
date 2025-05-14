import pandas as pd

def RSI(data, window=14):
    delta = data['Close'].diff(1)
    delta.index = data.index
    
    # 상승폭과 하락폭 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 평균 Gain과 평균 Loss 계산 (지수 이동평균 사용)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # RS 계산
    rs = avg_gain / avg_loss
    
    # RSI 계산
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT/BTCUSDT-1h-2023.csv', index_col=0)
    # RSI 계산
    rsi_values = RSI(data)
    print(rsi_values)