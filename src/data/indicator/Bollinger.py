import pandas as pd

def Bollinger(data, window=20, num_std_dev=2):
    # 이동 평균 계산 (중앙 밴드)
    middle_band = data['Close'].rolling(window=window).mean()
    
    # 이동 표준 편차 계산
    std_dev = data['Close'].rolling(window=window).std()
    
    # 상단 밴드와 하단 밴드 계산
    upper_band = middle_band + (num_std_dev * std_dev)
    lower_band = middle_band - (num_std_dev * std_dev)
    
    # 밴드 폭 계산
    band_width = (upper_band - lower_band) / middle_band
    
    # 밴드 폭 변화율 계산
    band_width_change = band_width.pct_change() * 100
    
    # 결과 반환
    return pd.DataFrame({
        'Middle Band': middle_band,
        'Upper Band': upper_band,
        'Lower Band': lower_band,
        'Band Width': band_width,
        'Band Width Change': band_width_change
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # 볼린저 밴드 계산
    bollinger_bands = Bollinger(data)
    print(bollinger_bands)