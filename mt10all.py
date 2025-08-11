import numpy as np
import pandas as pd

# =========================
# 0) 데이터 로드/전처리
# =========================
file_name = "mt-10yr-new.csv"
df = pd.read_csv(file_name, header=None)
df.columns = [
    "Date", "Open", "High", "Low", "Close",
    "Index1", "Index2", "Index3", "Index4", "Index5",
    "Index6", "Index7", "Index8", "Index9", "Index10", "Index11"
]
# 날짜 정규화(시간 제거)
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

# (금일값 / 전일값 - 1) * 100
def pct_change_today_over_yday(s, cur, prev):
    return (s[cur] / s[prev].shift(1) - 1) * 100

# 시가 변동률 / Index1 변동률 / 전일종가
df["Open_Pct_Change"]   = pct_change_today_over_yday(df, "Open",  "Close")
df["Index6_Pct_Change"] = pct_change_today_over_yday(df, "Index6","Index6")
df["Prev_Close"]        = df["Close"].shift(1)

# 거래 계산에 필요한 행만 별도로 (첫 행 NaN 등 제외)
need_cols = ["Date","Open","High","Low","Close","Prev_Close","Open_Pct_Change","Index6_Pct_Change"]
df_bt = df.loc[:, need_cols].dropna().reset_index(drop=True).copy()

# =========================
# 유틸 (스탑로스 단가 계산)
#  - ratio_is_percent=False 이면 0.005 => 0.5%
#  - 틱 사이즈 0.1로 천장올림
# =========================
def ceil_to_nearest(x, step):
    return np.ceil(x / step) * step

def stop_amount(prev_close, ratio, ratio_is_percent=False, tick=0.1):
    amt = prev_close * (ratio*0.01 if ratio_is_percent else ratio)
    return ceil_to_nearest(amt, tick)

# =========================
# 거래내역 생성
#  - 열린구간: a < idx < b, d < idx < c
#  - 상승: opct > idx*rising_rate → BUY, opct < idx*rising_rate → SELL
#  - 하락: opct < idx*falling_rate → SELL, opct > idx*falling_rate → BUY
#  - 스탑로스: 매수 O - ceil(PrevClose*sl_buy, 0.1), 매도 O + ceil(PrevClose*sl_sell, 0.1)
# =========================
def generate_trades_log(
    data,
    rising_rate, falling_rate,
    a, b, d, c,
    sl_buy, sl_sell,
    ratio_is_percent=False, tick=0.1
):
    logs = []
    for Date, O, H, L, C, PrevC, opct, idx in data[["Date","Open","High","Low","Close","Prev_Close","Open_Pct_Change","Index6_Pct_Change"]].itertuples(index=False, name=None):
        side = None
        band = None
        adj  = None
        stop = np.nan
        stop_hit = False
        pnl = 0.0

        # 상승 밴드 (열린구간)
        if (idx > 0) and (a < idx < b):
            band = "RISING"
            adj = idx * rising_rate
            if opct > adj:  # BUY
                side = "BUY"
                stop = O - stop_amount(PrevC, sl_buy,  ratio_is_percent, tick)
                if round(L,3) <= round(stop,3):
                    stop_hit = True; pnl = (stop - O)
                else:
                    pnl = (C - O)
            elif opct < adj:  # SELL
                side = "SELL"
                stop = O + stop_amount(PrevC, sl_sell, ratio_is_percent, tick)
                if round(H,3) >= round(stop,3):
                    stop_hit = True; pnl = (O - stop)
                else:
                    pnl = (O - C)

        # 하락 밴드 (열린구간)
        elif (idx < 0) and (d < idx < c):
            band = "FALLING"
            adj = idx * falling_rate
            if opct < adj:  # SELL
                side = "SELL"
                stop = O + stop_amount(PrevC, sl_sell, ratio_is_percent, tick)
                if round(H,3) >= round(stop,3):
                    stop_hit = True; pnl = (O - stop)
                else:
                    pnl = (O - C)
            elif opct > adj:  # BUY
                side = "BUY"
                stop = O - stop_amount(PrevC, sl_buy,  ratio_is_percent, tick)
                if round(L,3) <= round(stop,3):
                    stop_hit = True; pnl = (stop - O)
                else:
                    pnl = (C - O)

        if side is None:
            continue

        logs.append({
            "Date": Date,
            "Band": band,
            "IndexChange(%)": idx,
            "OpenPct(%)": opct,
            "AdjThreshold(%)": adj,
            "Side": side,
            "Entry": O,
            "Close": C,
            "Low": L,
            "High": H,
            "PrevClose": PrevC,
            "Stop": stop,
            "StopHit": stop_hit,
            "PnL": float(pnl),
        })

    trades = pd.DataFrame(logs).sort_values("Date").reset_index(drop=True)
    return trades

# =========================
# 일별 손익(원본 CSV의 모든 날짜 포함) 만들기
#  - 거래 없는 날 DailyPnL=0
#  - CumPnL은 0을 포함해 연속 누적
# =========================
def make_daily_pnl_all_dates(df_all_dates, trades_df):
    # 원본 CSV의 모든 날짜
    all_dates = (
        df_all_dates[["Date"]]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if trades_df.empty:
        daily = all_dates.copy()
        daily["DailyPnL"] = 0.0
        daily["Trades"]   = 0
        daily["CumPnL"]   = 0.0
        return daily

    byday = (
        trades_df.groupby("Date", as_index=False)
        .agg(DailyPnL=("PnL", "sum"), Trades=("PnL", "size"))
    )

    # 모든 날짜와 LEFT JOIN → 거래 없는 날 NaN → 0
    daily = all_dates.merge(byday, on="Date", how="left")
    daily["DailyPnL"] = daily["DailyPnL"].fillna(0.0)
    daily["Trades"]   = daily["Trades"].fillna(0).astype(int)
    daily["CumPnL"]   = daily["DailyPnL"].cumsum()
    return daily

# =========================
# 실행부 (파라미터는 네가 확정한 값을 넣어!)
# =========================
if __name__ == "__main__":
    # 이미 구해둔 ‘최적/확정’ 값 넣기
    rising_rate  = 2.460
    falling_rate = 2.520

    # 밴드 (열린구간)
    a, b = 0.20, 9.37          # 상승: a < idx < b
    d, c = -5.50, -0.03        # 하락: d < idx < c

    # 스탑로스 비율 (0.005=0.5%), 틱=0.1
    sl_buy  = 0.005
    sl_sell = 0.005
    ratio_is_percent = False
    tick = 0.1

    # 1) 거래 로그
    trades_df = generate_trades_log(
        df_bt,
        rising_rate, falling_rate,
        a, b, d, c,
        sl_buy, sl_sell,
        ratio_is_percent=ratio_is_percent, tick=tick
    )

    # 2) 일별 손익 (CSV의 모든 날짜 포함)
    daily_df = make_daily_pnl_all_dates(df, trades_df)

    # 3) 엑셀 저장
    out_path = "mt_kosdaq_trades.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        trades_df.to_excel(writer, sheet_name="trades", index=False)
        daily_df.to_excel(writer, sheet_name="daily_pnl", index=False)

    print(f"엑셀 저장 완료: {out_path}")
    print(f"- trades 행수: {len(trades_df)}")
    print(f"- daily_pnl 행수(원본 날짜 수): {len(daily_df)}")
