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

def calculate_pct_change(df, col_name_current, col_name_previous):
    # (금일값 / 전일값 - 1) * 100
    return (df[col_name_current] / df[col_name_previous].shift(1) - 1) * 100

# 시가 변동률: (시가 / 전일 종가 - 1) * 100
df["Open_Pct_Change"] = calculate_pct_change(df, "Open", "Close")
# Index1 변동률: (금일 Index1 / 전일 Index1 - 1) * 100
df["Index1_Pct_Change"] = calculate_pct_change(df, "Index1", "Index1")

# NaN(첫 행 등) 제거
df = df.dropna(subset=["Open_Pct_Change", "Index1_Pct_Change"]).reset_index(drop=True)


# =========================
# 1) 밴드 기반 손익 계산 (벡터화)
#    - 상승: rising_min < idx < rising_max
#    - 하락: falling_min < idx < falling_max
#    - 0은 자연스럽게 제외(열린구간)
# =========================
def calculate_trades_band_vectorized(
    df,
    index_col, open_pct_col, open_col, close_col,
    rising_rate, falling_rate,
    rising_min, rising_max,
    falling_min, falling_max
):
    # 안전장치: 밴드 유효성
    assert rising_min < rising_max, "rising_min < rising_max 이어야 합니다."
    assert falling_min < falling_max, "falling_min < falling_max 이어야 합니다."

    idx = df[index_col].values
    open_pct = df[open_pct_col].values
    O = df[open_col].values
    C = df[close_col].values

    # 밴드 마스크 (겹침/중복 없음, 0도 자동 제외)
    m_rise = (idx > rising_min) & (idx < rising_max)
    m_fall = (idx > falling_min) & (idx < falling_max)

    # 기준선
    adj_up = idx * rising_rate
    adj_dn = idx * falling_rate

    # 상승 밴드 손익 (기존 분기 로직 그대로)
    # open_pct > adj_up → C - O,  open_pct < adj_up → O - C (같으면 0)
    p_rise = (
        np.where(m_rise & (open_pct > adj_up), C - O, 0.0)
        + np.where(m_rise & (open_pct < adj_up), O - C, 0.0)
    ).sum()

    # 하락 밴드 손익
    # open_pct < adj_dn → O - C,  open_pct > adj_dn → C - O (같으면 0)
    p_fall = (
        np.where(m_fall & (open_pct < adj_dn), O - C, 0.0)
        + np.where(m_fall & (open_pct > adj_dn), C - O, 0.0)
    ).sum()

    return float(p_rise), float(p_fall)


# =========================
# 2) 밴드 최적화 (상승/하락 따로)
#    - 상승 밴드: (a, b) 탐색, a < b, a~b 범위는 필요에 맞게 조정
#    - 하락 밴드: (c, d) 탐색, d < c 가 아니라 여기서는 일관되게 (falling_min, falling_max)로 d < c 대신 (min < max) 규칙 사용
#      ※ Index1_Pct_Change는 음수도 나와서, 하락 밴드 범위는 음수 영역에서 탐색 추천
# =========================
def find_optimal_rising_band(
    df, index_col, open_pct_col, open_col, close_col, rising_rate, falling_rate,
    a_start=0.00, a_end=5.00, a_step=0.01,
    b_start=0.00, b_end=20.00, b_step=0.01,
    falling_min=-20.0, falling_max=-0.0  # 하락 밴드는 고정/무시(상승 최적만 볼 때)
):
    best = (-np.inf, None, None)  # (profit, a, b)
    idx_vals = df[index_col].values

    # 필요시 가속: 유효 데이터만 사전 필터링 가능
    # 여기서는 단순/명확성 우선
    a_grid = np.arange(a_start, a_end + 1e-12, a_step)
    b_grid = np.arange(b_start, b_end + 1e-12, b_step)

    for a in a_grid:
        # b는 반드시 a보다 커야 함
        valid_b = b_grid[b_grid > a]
        for b in valid_b:
            p_rise, _ = calculate_trades_band_vectorized(
                df, index_col, open_pct_col, open_col, close_col,
                rising_rate, falling_rate,
                rising_min=a, rising_max=b,
                falling_min=falling_min, falling_max=falling_max  # 하락 밴드는 영향 안 줌
            )
            if p_rise > best[0]:
                best = (p_rise, a, b)

    return best  # (best_profit, best_a, best_b)


def find_optimal_falling_band(
    df, index_col, open_pct_col, open_col, close_col, rising_rate, falling_rate,
    d_start=-20.0, d_end=0.00, d_step=0.01,   # falling_min 후보
    c_start=-5.00, c_end=0.00, c_step=0.01,    # falling_max 후보
    rising_min=0.00, rising_max=20.0           # 상승 밴드는 고정/무시(하락 최적만 볼 때)
):
    best = (-np.inf, None, None)  # (profit, d, c)
    d_grid = np.arange(d_start, d_end + 1e-12, d_step)  # 더 작은 음수 → 더 왼쪽
    c_grid = np.arange(c_start, c_end + 1e-12, c_step)  # 작은 음수 ~ 0

    for d in d_grid:
        # c는 반드시 d보다 커야 함 (예: -0.5 < c < 0, -20 < d < -0.5, 그리고 d < c 성립)
        valid_c = c_grid[c_grid > d]
        for c in valid_c:
            _, p_fall = calculate_trades_band_vectorized(
                df, index_col, open_pct_col, open_col, close_col,
                rising_rate, falling_rate,
                rising_min=rising_min, rising_max=rising_max,  # 상승 밴드 영향 안 줌
                falling_min=d, falling_max=c
            )
            if p_fall > best[0]:
                best = (p_fall, d, c)

    return best  # (best_profit, best_d, best_c)


# =========================
# 3) 실행 (예시 파라미터)
# =========================
index_col    = "Index1_Pct_Change"
open_pct_col = "Open_Pct_Change"
open_col     = "Open"
close_col    = "Close"

rising_rate  = 0.585
falling_rate = 0.920

# (1) 상승 밴드 최적화 (하락 밴드는 고정/무시 값으로)
best_rise_profit, best_rise_min, best_rise_max = find_optimal_rising_band(
    df, index_col, open_pct_col, open_col, close_col,
    rising_rate, falling_rate,
    a_start=0.00, a_end=5.00, a_step=0.01,   # 필요시 범위/스텝 조정
    b_start=0.00, b_end=20.00, b_step=0.01
)

# (2) 하락 밴드 최적화 (상승 밴드는 고정/무시 값으로)
best_fall_profit, best_fall_min, best_fall_max = find_optimal_falling_band(
    df, index_col, open_pct_col, open_col, close_col,
    rising_rate, falling_rate,
    d_start=-20.0, d_end=0.00, d_step=0.01,
    c_start=-5.00, c_end=0.00,  c_step=0.01
)

# (3) 최적 밴드로 최종 손익 계산 (동시에 적용해 보고 싶으면 여기서 한번 더)
final_rise, final_fall = calculate_trades_band_vectorized(
    df, index_col, open_pct_col, open_col, close_col,
    rising_rate, falling_rate,
    rising_min=best_rise_min, rising_max=best_rise_max,
    falling_min=best_fall_min, falling_max=best_fall_max
)

print(f"[상승 밴드 최적화] profit={best_rise_profit:.2f}, rising_min={best_rise_min:.2f}, rising_max={best_rise_max:.2f}")
print(f"[하락 밴드 최적화] profit={best_fall_profit:.2f}, falling_min={best_fall_min:.2f}, falling_max={best_fall_max:.2f}")
print(f"[최종 적용] 상승={final_rise:.2f}, 하락={final_fall:.2f}, 합계={(final_rise+final_fall):.2f}")
