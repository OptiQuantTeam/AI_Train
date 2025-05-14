import pandas as pd
import numpy as np

def MACD(data, short_window=12, long_window=26, signal_window=9, cross=True, divergence_lookback=20):
    # 짧은 기간과 긴 기간의 지수 이동 평균 계산
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    # MACD Line 계산
    macd_line = ema_short - ema_long
    
    # Signal Line 계산
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # Histogram 계산
    macd_histogram = round(macd_line - signal_line, 2)
    
    # Cross Signal 계산 (상향 돌파: 1, 하향 돌파: -1, 그 외: 0)
    cross_signal = np.zeros(len(data))
    
    for i in range(1, len(data)):
        # 상향 돌파: 이전에는 MACD < Signal, 현재는 MACD > Signal
        if macd_line.iloc[i-1] < signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
            cross_signal[i] = 1
        # 하향 돌파: 이전에는 MACD > Signal, 현재는 MACD < Signal
        elif macd_line.iloc[i-1] > signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
            cross_signal[i] = -1
    
    # 다이버전스 계산
    divergence_signal = np.zeros(len(data))
    
    for i in range(divergence_lookback, len(data)):
        # 가격과 MACD의 최근 고점/저점 찾기
        price_highs = []
        price_lows = []
        macd_highs = []
        macd_lows = []
        
        for j in range(i-divergence_lookback, i):
            # 가격 고점/저점 확인
            if (data.iloc[j]['High'] > data.iloc[j-1]['High'] and 
                data.iloc[j]['High'] > data.iloc[j+1]['High']):
                price_highs.append((j, data.iloc[j]['High']))
            if (data.iloc[j]['Low'] < data.iloc[j-1]['Low'] and 
                data.iloc[j]['Low'] < data.iloc[j+1]['Low']):
                price_lows.append((j, data.iloc[j]['Low']))
                
            # MACD 고점/저점 확인
            if (macd_line.iloc[j] > macd_line.iloc[j-1] and 
                macd_line.iloc[j] > macd_line.iloc[j+1]):
                macd_highs.append((j, macd_line.iloc[j]))
            if (macd_line.iloc[j] < macd_line.iloc[j-1] and 
                macd_line.iloc[j] < macd_line.iloc[j+1]):
                macd_lows.append((j, macd_line.iloc[j]))
        
        # 상승 다이버전스 확인
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            price_trend = price_lows[-1][1] - price_lows[-2][1]  # 가격 하락
            macd_trend = macd_lows[-1][1] - macd_lows[-2][1]    # MACD 상승
            if price_trend < 0 and macd_trend > 0:
                divergence_signal[i] = 1
                
        # 하락 다이버전스 확인
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            price_trend = price_highs[-1][1] - price_highs[-2][1]  # 가격 상승
            macd_trend = macd_highs[-1][1] - macd_highs[-2][1]    # MACD 하락
            if price_trend > 0 and macd_trend < 0:
                divergence_signal[i] = -1
    
    # 히스토그램 매매 신호 생성
    trade_signal = np.zeros(len(data))
    
    for i in range(1, len(data)):
        # 히스토그램이 0선을 상향 돌파
        if macd_histogram.iloc[i] > 0 and macd_histogram.iloc[i-1] <= 0:
            trade_signal[i] = 1
        # 히스토그램이 0선을 하향 돌파
        elif macd_histogram.iloc[i] < 0 and macd_histogram.iloc[i-1] >= 0:
            trade_signal[i] = -1
    
    # 결과 반환
    if cross:
        return macd_histogram
    else:
        return pd.DataFrame({
            'MACD Line': macd_line,
            'Signal Line': signal_line,
            'Histogram': macd_histogram,
            'Cross Signal': cross_signal,
            'Divergence Signal': divergence_signal,
            'Trade Signal': trade_signal
        }, index=data.index)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # MACD 계산
    macd_result = MACD(data, cross=False)
    print(macd_result)