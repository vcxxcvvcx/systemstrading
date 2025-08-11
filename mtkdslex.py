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
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()  # 시간 제거(정규화)

# (원본 모든 행 보존) 고유 행 식별자
df["RowId"] = np.arange(len(df))

def calculate_pct_change(df, col_name_current, col_name_previous):
    # (금일값 / 전일값 - 1) * 100
    return (df[col_name_current] / df[col_name_previous].shift(1) - 1) * 100

# 시가 변동률 & Index1 변동률
df["Open_Pct_Change"]   = calculate_pct_change(df, "Open",  "Close")
df["Index1_Pct_Change"] = calculate_pct_change(df, "Index1","Index1")

# 전일 종가
df["Prev_Close"] = df["Close"].shift(1)

# 백테스트용 데이터(필요 열 NaN 제거) — RowId 보존
df_bt = df.dropna(subset=["Open_Pct_Change", "Index1_Pct_Change", "Prev_Close"]).copy()

# 소수점 자리 맞추기 (예: 0.01 단위 상향 반올림)
def ceil_to_nearest(value, step):
    return np.ceil(value / step) * step


# =========================
# 1) 스탑로스 포함 손익 계산 (밴드 규칙)
#    - 상승: rising_min < idx < rising_max
#    - 하락: falling_min < idx < falling_max
# =========================
def calculate_filtered_trades_with_stop_loss(
    df, index_col, open_pct_col, open_col,
    rising_rate, falling_rate,
    rising_filter_min, rising_filter_max,
    falling_filter_min, falling_filter_max,
    stop_loss_buy_ratio, stop_loss_sell_ratio
):
    profit = 0.0
    need = ["Open","Close","Low","High","Prev_Close", index_col, open_pct_col]
    data = df.loc[:, need].rename(columns={index_col:"idx", open_pct_col:"opct"})

    for O, C, L, H, prevC, idx, open_pct in data.itertuples(index=False, name=None):
        # 상승 밴드
        if idx > 0 and (rising_filter_min < idx < rising_filter_max):
            adj = idx * rising_rate
            if open_pct > adj:  # 매수
                stop = O - ceil_to_nearest(prevC * stop_loss_buy_ratio, 0.1)
                profit += (stop - O) if round(L,3) <= round(stop,3) else (C - O)
            elif open_pct < adj:  # 매도
                stop = O + ceil_to_nearest(prevC * stop_loss_sell_ratio, 0.1)
                profit += (O - stop) if round(H,3) >= round(stop,3) else (O - C)

        # 하락 밴드
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
# 2) 스탑로스 최적화 (매수/매도 별도)
# =========================
def find_optimal_stop_loss(
    df, index_col, open_pct_col, open_col,
    rising_rate, falling_rate,
    rising_filter_min, rising_filter_max,
    falling_filter_min, falling_filter_max,
    stop_loss_type="buy"
):
    best_profit = -np.inf
    best_stop_loss = None

    # 0.05% ~ 5.0% (스텝 0.005%)
    stop_loss_range = np.arange(0.00050, 0.05000, 0.00005)

    for sl in stop_loss_range:
        if stop_loss_type == "buy":
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
# 3) 거래 로그 (개별 거래 기록, RowId 포함)
# =========================
def backtest_with_log(
    df, index_col, open_pct_col, open_col,
    rising_rate, falling_rate,
    rising_filter_min, rising_filter_max,
    falling_filter_min, falling_filter_max,
    stop_loss_buy_ratio, stop_loss_sell_ratio
):
    need = ["RowId","Date","Open","Close","Low","High","Prev_Close", index_col, open_pct_col]
    data = df.loc[:, need].rename(columns={index_col:"idx", open_pct_col:"opct"})

    logs = []
    for RowId, Date, O, C, L, H, prevC, idx, open_pct in data.itertuples(index=False, name=None):
        side = None
        stop = np.nan
        stop_hit = False
        pnl = 0.0
        band = None
        adj = None

        # 상승 밴드
        if idx > 0 and (rising_filter_min < idx < rising_filter_max):
            band = "RISING"
            adj = idx * rising_rate
            if open_pct > adj:  # BUY
                side = "BUY"
                stop = O - ceil_to_nearest(prevC * stop_loss_buy_ratio, 0.1)
                if round(L,3) <= round(stop,3):
                    stop_hit = True; pnl = (stop - O)
                else:
                    pnl = (C - O)
            elif open_pct < adj:  # SELL
                side = "SELL"
                stop = O + ceil_to_nearest(prevC * stop_loss_sell_ratio, 0.1)
                if round(H,3) >= round(stop,3):
                    stop_hit = True; pnl = (O - stop)
                else:
                    pnl = (O - C)

        # 하락 밴드
        elif idx < 0 and (falling_filter_min < idx < falling_filter_max):
            band = "FALLING"
            adj = idx * falling_rate
            if open_pct < adj:  # SELL
                side = "SELL"
                stop = O + ceil_to_nearest(prevC * stop_loss_sell_ratio, 0.1)
                if round(H,3) >= round(stop,3):
                    stop_hit = True; pnl = (O - stop)
                else:
                    pnl = (O - C)
            elif open_pct > adj:  # BUY
                side = "BUY"
                stop = O - ceil_to_nearest(prevC * stop_loss_buy_ratio, 0.1)
                if round(L,3) <= round(stop,3):
                    stop_hit = True; pnl = (stop - O)
                else:
                    pnl = (C - O)

        if side is None:
            continue

        logs.append({
            "RowId": RowId,
            "Date": Date,
            "Band": band,
            "IndexChange(%)": idx,
            "OpenPct(%)": open_pct,
            "AdjThreshold(%)": adj,
            "Side": side,
            "Entry": O,
            "Close": C,
            "Low": L,
            "High": H,
            "PrevClose": prevC,
            "Stop": stop,
            "StopHit": stop_hit,
            "PnL": float(pnl),
        })

    trades = pd.DataFrame(logs).sort_values("RowId")
    # 날짜 컬럼 정규화(혹시 모를 시간값 대비)
    if not trades.empty:
        trades["Date"] = pd.to_datetime(trades["Date"]).dt.normalize()
    return trades


# =========================
# 4) 실행
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

    # 최적 스탑로스 탐색 (매수/매도 별도) — df_bt 사용
    stop_loss_buy_ratio, profit_buy = find_optimal_stop_loss(
        df_bt, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_type="buy"
    )
    stop_loss_sell_ratio, profit_sell = find_optimal_stop_loss(
        df_bt, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_type="sell"
    )

    print(f"최적 스탑로스 (매수): {stop_loss_buy_ratio:.3%}, 해당 수익: {profit_buy:.2f}")
    print(f"최적 스탑로스 (매도): {stop_loss_sell_ratio:.3%}, 해당 수익: {profit_sell:.2f}")

    # 최적 스탑로스 동시 적용한 최종 수익
    final_profit = calculate_filtered_trades_with_stop_loss(
        df_bt, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_buy_ratio=stop_loss_buy_ratio,
        stop_loss_sell_ratio=stop_loss_sell_ratio
    )
    print(f"최종 수익 (최적 필터 + 스탑로스 적용): {final_profit:.2f}")

    # 거래 로그 생성 (RowId 포함)
    trades_df = backtest_with_log(
        df_bt, index_col, open_pct_col, open_col,
        rising_rate, falling_rate,
        rising_filter_min, rising_filter_max,
        falling_filter_min, falling_filter_max,
        stop_loss_buy_ratio=stop_loss_buy_ratio,
        stop_loss_sell_ratio=stop_loss_sell_ratio
    )

    # --------------------------------------------
    # 원본 CSV의 모든 행 기준으로 병합 (거래 없으면 공란)
    # --------------------------------------------
    base = df[["RowId","Date"]].copy()  # CSV 전체 행

    # RowId 기준 병합 시 Date 충돌 방지: trades에서 Date는 버리고 RowId만 사용
    if not trades_df.empty:
        trades_unique = trades_df.drop_duplicates(subset=["RowId"], keep="first").drop(columns=["Date"])
    else:
        trades_unique = trades_df.copy()

    rows_df = base.merge(trades_unique, on="RowId", how="left")

    # merge 후 Date 컬럼 정리(보장)
    # rows_df에는 base의 Date만 남아 있음

    # 행 기준 누적손익: 거래 없는 행은 0으로 누적
    rows_df["CumPnL"] = rows_df["PnL"].fillna(0).cumsum()

    # --------------------------------------------
    # 일자 요약 시트 (선택)
    # --------------------------------------------
    if rows_df["PnL"].notna().any():
        daily_df = (
            rows_df.groupby("Date", as_index=False)["PnL"]
            .sum()
            .rename(columns={"PnL":"DailyPnL"})
            .sort_values("Date")
        )
        daily_df["CumPnL"] = daily_df["DailyPnL"].cumsum()
    else:
        daily_df = pd.DataFrame(columns=["Date","DailyPnL","CumPnL"])

    # --------------------------------------------
    # 엑셀 저장 (openpyxl)
    # --------------------------------------------
    out_path = "mt_kosdaq_trades.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        rows_df.to_excel(writer, sheet_name="rows", index=False)         # CSV와 동일한 행수 유지
        daily_df.to_excel(writer, sheet_name="daily_pnl", index=False)   # 날짜 요약

    print(f"엑셀 저장 완료: {out_path}")
