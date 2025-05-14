import pandas as pd

def Stochastic(data, k_window=14, d_window=3):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # 기간 내 최고가와 최저가 계산
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    # %K 계산
    percent_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # %D 계산 (%K의 이동 평균)
    percent_d = percent_k.rolling(window=d_window).mean()
    
    # 결과 반환
    return pd.DataFrame({
        '%K': percent_k,
        '%D': percent_d
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 고가, 저가, 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # Stochastic 계산
    stochastic_result = Stochastic(data)
    print(stochastic_result)