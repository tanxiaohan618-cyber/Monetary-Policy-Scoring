#!/opt/anaconda3/bin/python
"""下载文本因子预测所需的中国利率与宏观控制变量。

输出：macro_rates.csv；单个可选接口失败不会阻断其余数据。
短端包括国债3M/6M、SHIBOR O/N/1W/1M/3M；中长端保留1Y—10Y国债。
"""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "macro_rates.csv"
CHECKPOINT_CSV = ROOT / "macro_rates_yields_checkpoint.csv"
START_DATE = pd.Timestamp("2011-01-01")
END_DATE = pd.Timestamp("2025-12-31")
TARGET_CURVE = "中债国债收益率曲线"
TARGET_MATURITIES = {
    "3月": "yield_3m", "6月": "yield_6m", "1年": "yield_1y",
    "3年": "yield_3y", "5年": "yield_5y", "7年": "yield_7y",
    "10年": "yield_10y",
}
SHIBOR_TENORS = {
    "隔夜": "shibor_on", "1周": "shibor_1w",
    "1月": "shibor_1m", "3月": "shibor_3m",
}
MAX_RETRIES = 3
REQUEST_INTERVAL_SECONDS = 0.8


def retry_call(label, function):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return function()
        except Exception as error:
            last_error = error
            print(f"{label} 第{attempt}/{MAX_RETRIES}次失败：{error}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    raise RuntimeError(f"{label}下载失败") from last_error


def parse_cn_month(value):
    match = re.search(r"(\d{4})年?(\d{1,2})", str(value))
    if not match:
        return pd.NaT
    year, month = map(int, match.groups())
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)


def fetch_yields():
    """逐年获取国债曲线；旧断点缺短端时自动重新下载。"""
    if CHECKPOINT_CSV.exists():
        old = pd.read_csv(CHECKPOINT_CSV)
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        complete = (
            set(TARGET_MATURITIES.values()).issubset(old.columns)
            and not old.empty and old.date.min().year <= START_DATE.year
            and old.date.max() >= END_DATE
        )
        if complete:
            print(f"复用完整国债断点：{CHECKPOINT_CSV}")
            return old.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    frames = []
    for year in range(START_DATE.year, END_DATE.year + 1):
        start = max(START_DATE, pd.Timestamp(year, 1, 1))
        end = min(END_DATE, pd.Timestamp(year, 12, 31))
        print(f"下载国债收益率：{year}年")
        raw = retry_call(
            f"{year}年国债收益率",
            lambda a=start, b=end: ak.bond_china_yield(
                start_date=a.strftime("%Y%m%d"), end_date=b.strftime("%Y%m%d")
            ),
        )
        available = [c for c in TARGET_MATURITIES if c in raw.columns]
        if not available:
            raise ValueError(f"{year}年接口没有目标期限，字段为：{raw.columns.tolist()}")
        missing = sorted(set(TARGET_MATURITIES) - set(available))
        if missing:
            warnings.warn(f"{year}年国债曲线缺少期限：{missing}")
        yearly = raw.loc[raw["曲线名称"].eq(TARGET_CURVE), ["日期", *available]].copy()
        yearly = yearly.rename(columns={"日期": "date", **TARGET_MATURITIES})
        yearly["date"] = pd.to_datetime(yearly["date"], errors="coerce")
        for col in yearly.columns.drop("date"):
            yearly[col] = pd.to_numeric(yearly[col], errors="coerce")
        frames.append(yearly.dropna(subset=["date"]))
        checkpoint = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
        checkpoint.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")


def _normalise_rate_frame(raw, output_column):
    date_candidates = [c for c in raw.columns if str(c).lower() in {"date", "日期", "报告日"}]
    value_candidates = [c for c in raw.columns if str(c) in {"利率", "值", "最新利率"}]
    if not date_candidates:
        date_candidates = [c for c in raw.columns if "日期" in str(c)]
    if not value_candidates:
        value_candidates = [c for c in raw.columns if "利率" in str(c)]
    if not date_candidates or not value_candidates:
        raise ValueError(f"无法识别日期/利率列：{raw.columns.tolist()}")
    out = raw[[date_candidates[0], value_candidates[0]]].copy()
    out.columns = ["date", output_column]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[output_column] = pd.to_numeric(out[output_column], errors="coerce")
    return out.dropna().drop_duplicates("date", keep="last").sort_values("date")


def fetch_shibor():
    """逐期限下载SHIBOR；接口不提供某一期限时仅跳过该列。"""
    frames = []
    for indicator, output_column in SHIBOR_TENORS.items():
        try:
            print(f"下载短端利率：{output_column}")
            raw = retry_call(
                output_column,
                lambda x=indicator: ak.rate_interbank(
                    market="上海银行同业拆借市场", symbol="Shibor人民币", indicator=x
                ),
            )
            frames.append(_normalise_rate_frame(raw, output_column))
        except Exception as error:
            warnings.warn(f"跳过{output_column}：{error}")
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return frames


def prepare_monthly(raw, value_column, output_column):
    if not {"月份", value_column}.issubset(raw.columns):
        raise ValueError(f"{output_column}接口字段不匹配：{raw.columns.tolist()}")
    frame = raw[["月份", value_column]].copy()
    frame["reference_date"] = frame["月份"].map(parse_cn_month)
    frame[output_column] = pd.to_numeric(frame[value_column], errors="coerce")
    # 保守发布日期假设：参考月数据从下一月末才进入信息集。
    frame["available_date"] = frame["reference_date"] + pd.offsets.MonthEnd(1)
    return frame[["available_date", output_column]].dropna().drop_duplicates("available_date").sort_values("available_date")


def fetch_macro_frames():
    specs = [
        ("cpi_yoy", ak.macro_china_cpi, "全国-同比增长"),
        ("ppi_yoy", ak.macro_china_ppi, "当月同比增长"),
        ("pmi_manufacturing", ak.macro_china_pmi, "制造业-指数"),
        ("m2_yoy", ak.macro_china_money_supply, "货币和准货币(M2)-同比增长"),
    ]
    frames = []
    for output_column, function, value_column in specs:
        try:
            print(f"下载宏观指标：{output_column}")
            frames.append(prepare_monthly(retry_call(output_column, function), value_column, output_column))
        except Exception as error:
            warnings.warn(f"跳过{output_column}：{error}")
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return frames


def merge_data(rates, daily_frames, macro_frames):
    data = rates.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    for frame in daily_frames:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype("datetime64[ns]")
        data = data.merge(frame, on="date", how="left")
    for frame in macro_frames:
        frame = frame.copy()
        frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce").astype("datetime64[ns]")
        data = pd.merge_asof(data.sort_values("date"), frame.sort_values("available_date"), left_on="date", right_on="available_date", direction="backward").drop(columns="available_date")
    return data.sort_values("date").reset_index(drop=True)


def validate(data):
    if data.empty or "date" not in data:
        raise ValueError("最终数据为空或缺少date")
    if data["date"].duplicated().any():
        raise ValueError("最终数据包含重复日期")
    rate_cols = [c for c in data if c.startswith(("yield_", "shibor_"))]
    if len(rate_cols) < 2:
        raise ValueError(f"可用利率序列过少：{rate_cols}")


def main():
    print(f"下载区间：{START_DATE.date()}—{END_DATE.date()}")
    rates = fetch_yields()
    result = merge_data(rates, fetch_shibor(), fetch_macro_frames())
    result = result[result["date"].between(START_DATE, END_DATE)].reset_index(drop=True)
    validate(result)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"下载完成：{OUTPUT_CSV}")
    print(f"日期范围：{result.date.min()} 至 {result.date.max()}；{len(result)}行")
    print("字段：", result.columns.tolist())


if __name__ == "__main__":
    main()
