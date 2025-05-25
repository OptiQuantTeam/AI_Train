import pandas as pd
import data.indicator as ind
import requests
from datetime import datetime
import numpy as np
import os

def getTrainData(ticker='BTCUSDT', startYear=2017, endYear=2023, interval='1h', raw=True):
    
    if raw:
        data = pd.DataFrame()
        while startYear<=endYear:
            path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{startYear}.csv'
            

            tmp = pd.read_csv(path, index_col=0)
            tmp['RSI'] = ind.RSI(tmp)
            tmp['EMAF'] = ind.EMA(tmp, window=10)
            tmp = tmp[['Open','High','Low','Close','RSI','EMAF']]
            data = pd.concat([data, tmp])
            startYear += 1
    else:
        path = f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}.csv'
        data = pd.read_csv(path, index_col=0)
    return data


# 1분, 5분, 30분, 1시간 등의 데이터(현재 데이터)를 거래소로부터 가져온다.
def getCurrentData(symbol, interval='30m', limit=None):
    url = "https://api.binance.com/api/v3/klines"
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Base asset volume', 'Number of trades',\
                'Taker buy volume', 'Taker buy base asset volume', 'Ignore']
    

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": None,
        "endTime": None,
        "limit": limit
    }
    res = requests.get(url, params=params)
    value = res.json()

    df = pd.DataFrame(value, columns=columns)
    
    df['Open'] = df['Open'].astype('float')
    df['High'] = df['High'].astype('float')
    df['Low'] = df['Low'].astype('float')
    df['Close'] = df['Close'].astype('float')


    df['Open time'] = df['Open time'].astype('int')
    df['Open time'] = df['Open time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df['Close time'] = df['Close time'].astype('int')
    df['Close time'] = df['Close time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df = df.set_index('Open time')
    
    env_path = preprocess_weekly(df, symbol, interval)
    #df.to_csv(f'/workspace/data/preprocess/{symbol}/{symbol}-{interval}-weekly.csv')

    return env_path

def preprocess_weekly(data, ticker='BTCUSDT', interval='1d'):

    """하이킨 아시 캔들 계산"""
    # Heikin Ashi 캔들 계산
    data['ha_close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    data['ha_open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    data['ha_high'] = data[['High', 'Open', 'Close']].max(axis=1)
    data['ha_low'] = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # 캔들 특성 계산
    data['ha_body'] = abs(data['ha_close'] - data['ha_open'])
    data['ha_lower_wick'] = np.minimum(data['ha_open'], data['ha_close']) - data['ha_low']
    data['ha_upper_wick'] = data['ha_high'] - np.maximum(data['ha_open'], data['ha_close'])
    
    # 하이킨 아시 신호 생성 (1: 상승, 0: 중립, -1: 하락)
    data['ha_signal'] = 0
    data.loc[(data['ha_close'] > data['ha_open']) & 
             (data['ha_lower_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = 1
    data.loc[(data['ha_close'] < data['ha_open']) & 
             (data['ha_upper_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = -1


    data['ha_high_diff'] = data['ha_high'] - data['ha_high'].shift(1)
    data['ha_low_diff'] = data['ha_low'] - data['ha_low'].shift(1)
    data['ha_body_diff'] = data['ha_body'] - data['ha_body'].shift(1)

    """200 EMA 계산"""
    data['ema_200'] = data['Close'].ewm(span=9600).mean()    # 30분봉 기준 200일 (200 * 48)
    data['ema_200_signal'] = 0
    data.loc[data['Close'] > data['ema_200'], 'ema_200_signal'] = 1
    data.loc[data['Close'] < data['ema_200'], 'ema_200_signal'] = -1

    """Stochastic RSI 계산"""
    # RSI 계산
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic RSI 계산
    data['stoch_rsi'] = (data['RSI'] - data['RSI'].rolling(14).min()) / \
                        (data['RSI'].rolling(14).max() - data['RSI'].rolling(14).min())
    
    # Stochastic RSI 신호 생성 (1: 과매수, 0: 중립, -1: 과매도)
    data['stoch_signal'] = 0
    data.loc[data['stoch_rsi'] < 0.2, 'stoch_signal'] = -1  # 과매도
    data.loc[data['stoch_rsi'] > 0.8, 'stoch_signal'] = 1   # 과매수

    """볼린저 밴드 계산"""
    bollinger_bands = ind.Bollinger(data, window=20, num_std_dev=2)
    data['bb_middle'] = bollinger_bands['bb_middle']
    data['bb_upper'] = bollinger_bands['bb_upper']
    data['bb_lower'] = bollinger_bands['bb_lower']
    data['bb_width'] = bollinger_bands['bb_width']
    data['bb_width_change'] = bollinger_bands['bb_width_change']

    # MACD
    MACD = ind.MACD(data, cross=False)
    data['MACD'] = MACD['Histogram']
    data['MACD_Signal'] = MACD['Signal Line']
    data['Cross Signal'] = MACD['Cross Signal']
    data['Divergence Signal'] = MACD['Divergence Signal']
    data['Trade Signal'] = MACD['Trade Signal']

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ha_high_diff', 'ha_low_diff', 'ha_body_diff',
                'ema_200', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_change',
                'MACD', 'MACD_Signal', 'Cross Signal', 'Divergence Signal', 'Trade Signal']]
    
    # NaN 값 제거
    data = data.dropna()
    
    os.makedirs(f'/workspace/src/data/preprocess', exist_ok=True)
    os.makedirs(f'/workspace/src/data/preprocess/{ticker}', exist_ok=True)

    data.to_csv(f'/workspace/src/data/preprocess/{ticker}/{ticker}-{interval}-weekly.csv')
    return f'/workspace/src/data/preprocess/{ticker}/{ticker}-{interval}-weekly.csv'
#timestamp = 1685577600000
#23년 6월 1일 오전 9시의 타임스탬프
#timestamp = 1732792920000
if __name__ == '__main__':
   
    df_path =  getCurrentData("BTCUSDT", "30m", limit=336)
    print(df_path)
    #df.to_csv("./BTCUSDT-15m-"+str(year)+".csv")
    
