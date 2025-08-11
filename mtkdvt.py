import numpy as np
import pandas as pd

# Step 1: 업로드된 파일 읽기
file_name = "mt-kosdaq-new.csv"
df = pd.read_csv(file_name, header=None)

# Step 2: 데이터 전처리
df.columns = [
    "Date", "Open", "High", "Low", "Close",
    "Index1", "Index2", "Index3", "Index4", "Index5",
    "Index6", "Index7", "Index8", "Index9", "Index10", "Index11"
]
df["Date"] = pd.to_datetime(df["Date"])

def calculate_pct_change(df, col_name_current, col_name_previous):
    return (df[col_name_current] / df[col_name_previous].shift(1) - 1) * 100

df["Open_Pct_Change"] = calculate_pct_change(df, "Open", "Close")

for i in range(1, 12):
    df[f"Index{i}_Pct_Change"] = calculate_pct_change(df, f"Index{i}", f"Index{i}")

# =========================
# 벡터화된 최적 조정률 탐색
# - 조정률 그리드를 한 번에 평가 (브로드캐스팅)
# - 원래 로직의 부등호/분기 그대로 유지
# =========================
def find_optimal_adjustment_rate(index_col, open_pct_col, open_col, df,
                                 rate_min=0.000, rate_max=5.000, rate_step=0.005):
    # 필요한 컬럼만 사용 + NaN 제거
    data = df[[index_col, open_pct_col, "Open", "Close"]].dropna().copy()

    idx   = data[index_col].to_numpy(dtype=np.float64)      # 전일 대비 지수 변동률 (%)
    opct  = data[open_pct_col].to_numpy(dtype=np.float64)   # (시가/전일종가 - 1)*100
    O     = data["Open"].to_numpy(dtype=np.float64)
    C     = data["Close"].to_numpy(dtype=np.float64)

    up    = (idx > 0)    # 상승일
    down  = (idx < 0)    # 하락일

    # 조정률 그리드 (R,)
    rates = np.arange(rate_min, rate_max, rate_step, dtype=np.float64)
    if rates.size == 0:
        raise ValueError("rate range is empty. check min/max/step.")

    # (R, N)로 브로드캐스팅
    adj = np.outer(rates, idx)            # adjusted_index_change = idx * rate
    op  = opct[None, :]                   # (1, N)
    O_mat = O[None, :]
    C_mat = C[None, :]

    # 상승 구간 수익: up 마스크 안에서만 계산
    # if op > adj: C - O
    # elif op < adj: O - C
    # else: 0
    up_mask = up[None, :]                 # (1, N)
    rise_profit = np.where(up_mask & (op > adj),  C_mat - O_mat, 0.0) \
                + np.where(up_mask & (op < adj),  O_mat - C_mat, 0.0)
    rise_profit_sum = rise_profit.sum(axis=1)     # (R,)

    # 하락 구간 수익: down 마스크 안에서만 계산
    # if op < adj: O - C
    # elif op > adj: C - O
    # else: 0
    down_mask = down[None, :]
    fall_profit = np.where(down_mask & (op < adj),  O_mat - C_mat, 0.0) \
                + np.where(down_mask & (op > adj),  C_mat - O_mat, 0.0)
    fall_profit_sum = fall_profit.sum(axis=1)     # (R,)

    # 최적값 선택
    r_idx = int(np.argmax(rise_profit_sum))
    f_idx = int(np.argmax(fall_profit_sum))

    best_rising_rate   = float(rates[r_idx])
    best_rising_profit = float(rise_profit_sum[r_idx])
    best_falling_rate  = float(rates[f_idx])
    best_falling_profit= float(fall_profit_sum[f_idx])

    return best_rising_rate, best_rising_profit, best_falling_rate, best_falling_profit


# 각 인덱스에 대해 계산
optimal_rates = {}
for i in range(1, 12):
    index_col = f"Index{i}_Pct_Change"
    open_pct_col = "Open_Pct_Change"
    open_col = "Open"

    rising_rate, rising_profit, falling_rate, falling_profit = find_optimal_adjustment_rate(
        index_col, open_pct_col, open_col, df,
        rate_min=0.000, rate_max=5.000, rate_step=0.005
    )
    optimal_rates[f"Index{i}"] = {
        "Rising Optimal Rate": rising_rate,
        "Rising Max Profit": rising_profit,
        "Falling Optimal Rate": falling_rate,
        "Falling Max Profit": falling_profit
    }

# 결과 출력
for index, values in optimal_rates.items():
    print(f"{index}:")
    print(f"  최적 상승 조정률 = {values['Rising Optimal Rate']:.3f}, 최대 상승 수익 = {values['Rising Max Profit']:.2f}")
    print(f"  최적 하락 조정률 = {values['Falling Optimal Rate']:.3f}, 최대 하락 수익 = {values['Falling Max Profit']:.2f}")
    print()
