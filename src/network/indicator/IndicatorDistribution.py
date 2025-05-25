import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IndicatorDistribution(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IndicatorDistribution, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 하이킨 아시 캔들 분석을 위한 네트워크
        self.ha_network = nn.Sequential(
            nn.Linear(7, 32),  # ha_close, ha_open, ha_high, ha_low, ha_body, ha_lower_wick, ha_upper_wick
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 200 MA 분석을 위한 네트워크
        self.ma_network = nn.Sequential(
            nn.Linear(2, 16),  # ma_200, ma_200_signal
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        
        # Stochastic RSI 분석을 위한 네트워크
        self.stoch_network = nn.Sequential(
            nn.Linear(2, 16),  # stoch_rsi, stoch_signal
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

        # 볼린저 밴드 분석을 위한 네트워크
        self.bb_network = nn.Sequential(
            nn.Linear(3, 16),  # bb_upper, bb_middle, bb_lower
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

    def forward(self, state):
        batch_size = state.shape[0]
        
        # 1) 기본 확률 정의 - 더 균형잡힌 분포로 수정
        default_probs = torch.tensor([0.4, 0.2, 0.4], device=state.device)  # SHORT, HOLD, LONG
        default_logits = default_probs.unsqueeze(0).expand(batch_size, -1)
        
        # 2) 하이킨 아시 캔들 분석 - 추세 추종 강화
        ha_open = state[:, 0]     # ha_open
        ha_close = state[:, 1]    # ha_close
        ha_high = state[:, 2]     # ha_high
        ha_low = state[:, 3]      # ha_low
        ha_body = state[:, 4]     # ha_body
        ha_lower_wick = state[:, 5]  # ha_lower_wick
        ha_upper_wick = state[:, 6]  # ha_upper_wick
        
        
        # 현재와 이전 캔들의 관계
        high_diff = state[:, 8]
        low_diff = state[:, 9]
        body_diff = state[:, 10]
        
        # 캔들 패턴 분석 - 추세 추종 강화
        is_bullish = ha_close > ha_open  # 양봉
        is_bearish = ha_close < ha_open  # 음봉
        body_size = torch.abs(ha_close - ha_open)  # 몸통 크기
        has_small_lower_wick = ha_lower_wick < 1e-6  # 작은 아래꼬리
        has_small_upper_wick = ha_upper_wick < 1e-6  # 작은 위꼬리
        
        # 연속 캔들 패턴 확인 (추세 강화)
        trend_bullish = torch.logical_and(is_bullish, body_diff > 0)  # 상승 추세
        trend_bearish = torch.logical_and(is_bearish, body_diff < 0)  # 하락 추세
        
        ha_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 강한 상승 신호 - 추세 추종 강화
        strong_bullish = torch.logical_and(
            torch.logical_and(trend_bullish, body_size > 0.5),
            torch.logical_and(high_diff > 0, low_diff > 0)  # 고가와 저가 모두 상승
        )
        ha_signal_tensor[strong_bullish, 2] = 0.7  # LONG 가중치 조정
        
        # 강한 하락 신호 - 추세 추종 강화
        strong_bearish = torch.logical_and(
            torch.logical_and(trend_bearish, body_size > 0.5),
            torch.logical_and(high_diff < 0, low_diff < 0)  # 고가와 저가 모두 하락
        )
        ha_signal_tensor[strong_bearish, 0] = 0.7  # SHORT 가중치 조정
        
        # 3) 200 MA 분석 - 추세 추종 강화
        ma_200 = state[:, 11]        # ma_200
        ma_200_signal = state[:, 12]  # ma_200_signal
        
        # MA 기울기 계산 (추세 강도)
        ma_slope = (ma_200 - torch.roll(ma_200, 1, 0)) / ma_200
        
        # 4) Stochastic RSI 분석 - 과매수/과매도 구간 조정
        stoch_rsi = state[:, 13]      # stoch_rsi
        stoch_signal = state[:, 14]   # stoch_signal
        
        # 5) 볼린저 밴드 분석 - 추세 추종 강화
        bb_middle = state[:, 15]     # bb_middle
        bb_upper = state[:, 16]      # bb_upper
        bb_lower = state[:, 17]      # bb_lower
        
        bb_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 볼린저 밴드 신호 분석 - 추세 추종 강화
        bb_width = state[:, 18]
        price_position = (ha_close - bb_lower) / (bb_upper - bb_lower)  # 가격 위치
        
        # 밴드 수축/확장 상태
        band_squeeze = bb_width < 0.1
        band_expansion = bb_width > 0.2
        
        # 볼린저 밴드 기반 신호 - 추세 추종 강화
        bb_signal_tensor[torch.logical_and(price_position < 0.2, ma_slope > 0), 2] = 0.7  # 하단 밴드 + 상승 추세 -> LONG
        bb_signal_tensor[torch.logical_and(price_position > 0.8, ma_slope < 0), 0] = 0.7  # 상단 밴드 + 하락 추세 -> SHORT
        
        # 밴드 수축 후 확장 시 신호
        squeeze_expansion = torch.logical_and(band_squeeze, band_expansion)
        bb_signal_tensor[torch.logical_and(squeeze_expansion, ma_slope > 0), 2] = 0.8  # 수축 후 확장 + 상승 추세 -> LONG
        bb_signal_tensor[torch.logical_and(squeeze_expansion, ma_slope < 0), 0] = 0.8  # 수축 후 확장 + 하락 추세 -> SHORT
        
        stoch_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # Stochastic RSI 과매수/과매도 구간 - 추세 고려
        overbought = torch.logical_and(stoch_rsi > 0.8, ma_slope < 0)  # 과매수 + 하락 추세"
        oversold = torch.logical_and(stoch_rsi < 0.2, ma_slope > 0)    # 과매도 + 상승 추세
        
        # Stochastic RSI 신호
        stoch_signal_tensor[oversold, 2] = 0.7  # 과매도 + 상승 추세 -> LONG
        stoch_signal_tensor[overbought, 0] = 0.7  # 과매수 + 하락 추세 -> SHORT
        
        # 200선 신호 - 추세 추종 강화
        ma_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        ma_signal_tensor[torch.logical_and(ma_200_signal > 0.1, ma_slope > 0), 2] = 0.7  # 상단 + 상승 추세 -> LONG
        ma_signal_tensor[torch.logical_and(ma_200_signal < -0.1, ma_slope < 0), 0] = 0.7  # 하단 + 하락 추세 -> SHORT
        
        # 6) 통합 신호 생성 - 추세 추종 강화
        ha_ma_stoch_bb_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 롱 진입 신호 - 추세 추종 강화
        long_signal = (
            (torch.logical_and(ha_signal_tensor[:, 2] > 0, ma_slope > 0)) &  # 하이킨 아시 + 상승 추세
            (
                (torch.logical_and(ma_200_signal > 0.1, stoch_signal < -0.1)) |  # MA + RSI
                (torch.logical_and(ma_200_signal > 0.1, price_position < 0.2)) |  # MA + BB
                (torch.logical_and(stoch_signal < -0.1, price_position < 0.2))  # RSI + BB
            )
        )
        
        # 숏 진입 신호 - 추세 추종 강화
        short_signal = (
            (torch.logical_and(ha_signal_tensor[:, 0] > 0, ma_slope < 0)) &  # 하이킨 아시 + 하락 추세
            (
                (torch.logical_and(ma_200_signal < -0.1, stoch_signal > 0.1)) |  # MA + RSI
                (torch.logical_and(ma_200_signal < -0.1, price_position > 0.8)) |  # MA + BB
                (torch.logical_and(stoch_signal > 0.1, price_position > 0.8))  # RSI + BB
            )
        )
        
        ha_ma_stoch_bb_signal[long_signal, 2] = 0.8  # LONG 가중치
        ha_ma_stoch_bb_signal[short_signal, 0] = 0.8  # SHORT 가중치
        
        # 7) 모든 신호 통합 - 가중치 조정
        combined_signal = (
            ha_signal_tensor * 1.2 +  # 하이킨 아시
            ma_signal_tensor * 1.5 +  # MA (추세 지표 가중치 증가)
            stoch_signal_tensor * 1.0 +  # RSI
            bb_signal_tensor * 1.2 +  # 볼린저 밴드
            ha_ma_stoch_bb_signal * 1.8  # 통합 신호
        )
        
        # 8) 최종 확률 분포 계산
        mixed_logits = default_logits + combined_signal
        final_probs = F.softmax(mixed_logits / 0.5, dim=-1)
        
        
        return final_probs
        