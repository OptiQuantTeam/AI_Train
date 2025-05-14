import pandas as pd


def _TR(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range

def KeltnerChannel(data, ema_window=20, atr_window=14, atr_multiplier=2):
    high = data['High']
    low = data['Low']
    close = data['Close']
    # EMA 계산 (중심선)
    middle_line = close.ewm(span=ema_window, adjust=False).mean()
    
    # ATR 계산
    true_range = _TR(high, low, close)
    atr = true_range.rolling(window=atr_window).mean()
    
    # 상단 밴드와 하단 밴드 계산
    upper_band = middle_line + (atr * atr_multiplier)
    lower_band = middle_line - (atr * atr_multiplier)
    
    # 결과 반환
    return pd.DataFrame({
        'Middle Line': middle_line,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 고가, 저가, 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # Keltner Channel 계산
    keltner_channel = KeltnerChannel(data)
    print(keltner_channel)
