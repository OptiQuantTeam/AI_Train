import pandas as pd

def Momentum(data, window=14):
    # 현재 가격과 과거 가격 차이를 계산
    momentum = data['Close'] - data['Close'].shift(window)
    
    # 모멘텀 백분율 계산
    momentum_percentage = ((data['Close'] / data['Close'].shift(window)) - 1) * 100
    
    return pd.DataFrame({
        'Momentum': momentum,
        'Momentum (%)': momentum_percentage
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # 모멘텀 계산
    momentum_result = Momentum(data, window=5)
    print(momentum_result)
