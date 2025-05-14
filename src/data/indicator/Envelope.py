import pandas as pd

def Envelope(data, window=20, deviation=0.02, method='SMA'):

    # 이동 평균 계산
    if method == 'SMA':
        ma = data['Close'].rolling(window=window).mean()
    elif method == 'EMA':
        ma = data['Close'].ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("Invalid method. Choose 'SMA' or 'EMA'.")
    
    # 상단 밴드와 하단 밴드 계산
    upper_band = ma * (1 + deviation)
    lower_band = ma * (1 - deviation)
    
    # 결과 반환
    return pd.DataFrame({
        'Moving Average': ma,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # Envelope 계산
    envelope_result = Envelope(data, deviation=0.02, method='SMA')
    print(envelope_result)
