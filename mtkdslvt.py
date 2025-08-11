import numpy as np
import pandas as pd

# =========================
# 0) 데이터 로드/전처리
# =========================
file_name = "mt-kosdaq-new.csv"
df = pd.read_csv(file_name, header=None)

df.columns = [
    "Date", "Open", "High", "Low", "Close",
    "Index1", "Index2", "Index3", "Index4", "Index5",
    "Index6", "Index7", "Index8", "Index9", "Index10", "Index11"
]
df["Date"] = pd.to_datetime(df["Date"])

# (금일값 / 전일값 - 1) * 100
def calculate_pct_change(df, col_name_current, col_name_previous):
    return (df[col_name_current] / df[col_name_previous].shift(1) - 1) * 100

# 시가 변동률: (시가 / 전일 종가 - 1) * 100
df["Open_Pct_Change"] = calculate_pct_change(df, "Open", "Close")
# Index1 변동률: (금일 Index1 / 전일 Index1 - 1) * 100
df["Index1_Pct_Change"] = calculate_pct_change(df, "Index1", "Index1")

# 전일 종가
df["Prev_Close"] = df["Close"].shift(1)

# 소수점 자리 맞추기 (예: 0.01 단위 상향 반올림)
def ceil_to_nearest(value, step):
    return np.ceil(value / step) * step


# =========================
# 1) 스탑로스 포함 손익 계산 (itertuples)
#    - 상승: rising_min < idx < rising_max
#    - 하락: falling_min < idx < falling_max
#    - 0은 자동 제외 (열린구간)
# =========================
def calculate_filtered_trades_with_stop_loss(df, index_col, open_pct_col, open_col,
                                             rising_rate, falling_rate,
                                             rising_filter_min, rising_filter_max,
                                             falling_filter_min, falling_filter_max,
                                             stop_loss_buy_ratio, stop_loss_sell_ratio):
    profit = 0.0

    # 필요한 컬럼만, NaN 제거 (Prev_Close 첫 행 등)
    need = ["Open","Close","Low","High","Prev_Close", index_col, open_pct_col]
    data = df.loc[:, need].dropna(subset=["Prev_Close"]).copy()

    # 필드명 단순화 (namedtuple 속성 접근 편하게)
    data = data.rename(columns={index_col: "idx", open_pct_col: "opct"})

    for O, C, L, H, prevC, idx, open_pct in data.itertuples(index=False, name=None):
        # 상승 밴드: (rising_min, rising_max) 열린구간
        if idx > 0 and (rising_filter_min < idx < rising_filter_max):
            adj = idx * rising_rate
            if open_pct > adj:  # 매수
                stop = O - ceil_to_nearest(prevC * stop_loss_buy_ratio, 0.1)
                profit += (stop - O) if round(L,3) <= round(stop,3) else (C - O)
            elif open_pct < adj:  # 매도
                stop = O + ceil_to_nearest(prevC * stop_loss_sell_ratio, 0.1)
                profit += (O - stop) if round(H,3) >= round(stop,3) else (O - C)

        # 하락 밴드: (falling_min, falling_max) 열린구간
        elif idx < 0 and (falling_filter_min < idx < falling_filter_max):
            adj = idx * falling_rate
            if open_pct < adj:  # 매도
                stop = O + ceil_to_nearest(prevC * stop_loss_sell_ratio, 0.1)
                profit += (O - stop) if round(H,3) >= round(stop,3) else (O - C)
            elif open_pct > adj:  # 매수
                stop = O - ceil_to_nearest(prevC * stop_loss_buy_ratio, 0.1)
                profit += (stop - O) if round(L,3) <= round(stop,3) else (C - O)

    return float(profit)


# =========================
# 2) 스탑로스 최적화 (매수/매도 따로)
# =========================
def find_optimal_stop_loss(df, index_col, open_pct_col, open_col,
                           rising_rate, falling_rate,
                           rising_filter_min, rising_filter_max,
                           falling_filter_min, falling_filter_max,
                           stop_loss_type='buy'):
    best_profit = -np.inf
    best_stop_loss = None

    # 0.05% ~ 5.0% 범위 (스텝 0.005%)
    stop_loss_range = np.arange(0.00050, 0.05000, 0.00005)

    for sl in stop_loss_range:
        if stop_loss_type == 'buy':
            profit = calculate_filtered_trades_with_stop_loss(
                df, index_col, open_pct_col, open_col,
                rising_rate, falling_rate,
                rising_filter_min, rising_filter_max,
                falling_filter_min, falling_filter_max,
                stop_loss_buy_ratio=sl, stop_loss_sell_ratio=0.0
            )
        else:  # 'sell'
            profit = calculate_filtered_trades_with_stop_loss(
                df, index_col, open_pct_col, open_col,
                rising_rate, falling_rate,
                rising_filter_min, rising_filter_max,
                falling_filter_min, falling_filter_max,
                stop_loss_buy_ratio=0.0, stop_loss_sell_ratio=sl
            )

        if profit > best_profit:
            best_profit = profit
            best_stop_loss = sl

    return best_stop_loss, best_profit


# =========================
# 3) 실행
# =========================
if __name__ == "__main__":
    index_col    = "Index1_Pct_Change"
    open_pct_col = "Open_Pct_Change"
    open_col     = "Open"

    rising_rate  = 0.585
    falling_rate = 0.920

    # (엑셀/사전 계산된) 최적 필터
    rising_filter_min  = 0.20
    rising_filter_max  = 9.37
    falling_filter_min = -5.50
    falling_filter_max = -0.03

    # 최적 스탑로스 탐색 (매수/매도 별도)
    stop_loss_buy_ratio, profit_with_stop_loss_buy = find_optimal_stop_loss(
        df, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_type='buy'
    )

    stop_loss_sell_ratio, profit_with_stop_loss_sell = find_optimal_stop_loss(
        df, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_type='sell'
    )

    print(f"최적 스탑로스 (매수): {stop_loss_buy_ratio:.3%}, 해당 수익: {profit_with_stop_loss_buy:.2f}")
    print(f"최적 스탑로스 (매도): {stop_loss_sell_ratio:.3%}, 해당 수익: {profit_with_stop_loss_sell:.2f}")

    # 최적 스탑로스 동시 적용한 최종 수익
    final_profit = calculate_filtered_trades_with_stop_loss(
        df, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_buy_ratio=stop_loss_buy_ratio,
        stop_loss_sell_ratio=stop_loss_sell_ratio
    )
    print(f"최종 수익 (최적 필터 + 스탑로스 적용): {final_profit:.2f}")
