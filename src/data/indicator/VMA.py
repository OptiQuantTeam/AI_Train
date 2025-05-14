import pandas as pd

def VMA(data, window=20, method='SMA'):
    if method == 'SMA':
        vma = data['Volume'].rolling(window=window).mean()
    elif method == 'EMA':
        vma = data['Volume'].ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("Invalid method. Choose 'SMA' or 'EMA'.")
    
    return vma

# 예제 데이터 사용
if __name__ == "__main__":
    # 거래량 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # 거래량 이동평균선 계산
    sma_vma = VMA(data, window=5, method='SMA')
    ema_vma = VMA(data, window=5, method='EMA')
    
    print("SMA VMA:\n", sma_vma)
    print("EMA VMA:\n", ema_vma)
