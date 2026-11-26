"""
US Market EOD & True Intraday VWAP Screener Data Generator
==========================================================
Fetches NYSE & NASDAQ common stocks, evaluates your 7-rule breakout scan across
the last 9 months (189 trading days), computes TRUE intraday VWAP from 5-minute
bars, and exports `us_eod_data.json` for the Chartink-style US dashboard.

Requirements:
    pip install yfinance pandas numpy requests

Usage:
    python fetch_us_eod.py
    python fetch_us_eod.py --output us_eod_data.json --days 189
"""

import argparse
import io
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# Map Yahoo Finance GICS sector names to the exact 11 keys in the web dashboard
SECTOR_MAP = {
    "Technology": "technology",
    "Healthcare": "healthcare",
    "Financial Services": "financials",
    "Consumer Cyclical": "consumer discretionary",
    "Industrials": "industrials",
    "Communication Services": "communication services",
    "Energy": "energy",
    "Consumer Defensive": "consumer staples",
    "Basic Materials": "materials",
    "Real Estate": "real estate",
    "Utilities": "utilities",
}


def fetch_us_common_stocks(max_symbols: int = 0) -> pd.DataFrame:
    """
    Downloads official NYSE & NASDAQ common stock lists with automatic
    cloud fallback (NASDAQ Trader -> US SEC EDGAR Exchange Directory) so
    GitHub Actions runners are never blocked by firewall rules.
    """
    print("[1/5] Fetching official NYSE & NASDAQ symbol lists...")
    universe = pd.DataFrame()

    # Source 1: NASDAQ Trader Symbol Directory
    try:
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/plain,text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        r_nas = requests.get(nasdaq_url, headers=browser_headers, timeout=20)
        r_nas.raise_for_status()
        df_nas = pd.read_csv(io.StringIO(r_nas.text), sep="|")
        df_nas = df_nas[df_nas["Symbol"].notna() & ~df_nas["Symbol"].str.contains("File Creation Time", na=False)]
        df_nas = df_nas[(df_nas["Test Issue"] == "N") & (df_nas["ETF"] == "N")].copy()
        df_nas["Exchange"] = "NASDAQ"
        df_nas = df_nas[["Symbol", "Security Name", "Exchange"]]

        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        r_oth = requests.get(other_url, headers=browser_headers, timeout=20)
        r_oth.raise_for_status()
        df_oth = pd.read_csv(io.StringIO(r_oth.text), sep="|")
        df_oth = df_oth[df_oth["ACT Symbol"].notna() & ~df_oth["ACT Symbol"].str.contains("File Creation Time", na=False)]
        df_oth = df_oth[(df_oth["Test Issue"] == "N") & (df_oth["ETF"] == "N")].copy()
        df_oth = df_oth[df_oth["Exchange"].isin(["N", "A", "P"])].copy()
        df_oth["Exchange"] = df_oth["Exchange"].map({"N": "NYSE", "A": "NYSE", "P": "NYSE"})
        df_oth = df_oth.rename(columns={"ACT Symbol": "Symbol"})[["Symbol", "Security Name", "Exchange"]]

        universe = pd.concat([df_nas, df_oth], ignore_index=True).drop_duplicates(subset=["Symbol"])
        print("      Loaded symbol list from NASDAQ Trader.")
    except Exception as exc:
        print(f"      NASDAQ Trader unavailable on cloud runner ({exc}). Switching to US SEC EDGAR...")

    # Source 2: Official US SEC EDGAR Company & Exchange Directory (Cloud-Safe Fallback)
    if universe.empty:
        sec_headers = {
            "User-Agent": "USMarketScreener/1.0 (contact@usmarketscreener.org)",
            "Accept": "application/json",
        }
        sec_url = "https://www.sec.gov/files/company_tickers_exchange.json"
        r_sec = requests.get(sec_url, headers=sec_headers, timeout=25)
        r_sec.raise_for_status()
        payload = r_sec.json()
        df_sec = pd.DataFrame(payload["data"], columns=payload["fields"])
        df_sec = df_sec[df_sec["exchange"].isin(["Nasdaq", "NYSE"])].copy()
        df_sec["Exchange"] = df_sec["exchange"].map({"Nasdaq": "NASDAQ", "NYSE": "NYSE"})
        df_sec = df_sec.rename(columns={"ticker": "Symbol", "name": "Security Name"})[
            ["Symbol", "Security Name", "Exchange"]
        ]
        universe = df_sec.drop_duplicates(subset=["Symbol"])
        print("      Loaded symbol list from US SEC EDGAR.")

    # Filter out warrants, rights, units, preferreds, and tickers with special characters
    def is_clean_common_ticker(sym: str) -> bool:
        if not isinstance(sym, str):
            return False
        sym = sym.strip()
        if any(ch in sym for ch in [".", "$", "^", "/", "-", " "]):
            return False
        # Exclude 5-letter NASDAQ suffixes for warrants/rights/units/bankrupt (W, R, U, Q)
        if len(sym) == 5 and sym[-1] in {"W", "R", "U", "Q"}:
            return False
        return 1 <= len(sym) <= 5

    universe = universe[universe["Symbol"].apply(is_clean_common_ticker)].reset_index(drop=True)
    universe["CleanName"] = universe["Security Name"].apply(
        lambda x: str(x).split(" - ")[0].strip() if pd.notna(x) else ""
    )

    if max_symbols > 0:
        universe = universe.head(max_symbols)

    print(f"      Found {len(universe):,} clean NYSE & NASDAQ common stocks.")
    return universe


def download_daily_ohlcv_batched(symbols: List[str], batch_size: int = 250) -> Dict[str, pd.DataFrame]:
    """
    Downloads ~11 months of daily OHLCV bars in batches so 20-day and 30-day EMAs
    are warmed up across a full 9-month (189 trading days) backtest window.
    """
    print(f"[2/5] Downloading 11-month daily OHLCV history in batches of {batch_size}...")
    daily_data: Dict[str, pd.DataFrame] = {}
    total_batches = math.ceil(len(symbols) / batch_size)

    for b_idx in range(total_batches):
        chunk = symbols[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"      Batch {b_idx + 1}/{total_batches} ({len(chunk)} tickers)...")
        try:
            raw = yf.download(
                tickers=chunk,
                period="11mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"      Warning: Batch {b_idx + 1} error: {exc}")
            continue

        if raw.empty:
            continue

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for sym in chunk:
            try:
                if is_multi:
                    if sym in raw.columns.get_level_values(0):
                        df = raw[sym].copy()
                    elif sym in raw.columns.get_level_values(1):
                        df = raw.xs(sym, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df = raw.copy()
                df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                df = df[df["Volume"] > 0]
                if len(df) >= 35:
                    daily_data[sym] = df
            except Exception:
                continue

        time.sleep(0.4)

    print(f"      Loaded valid daily history for {len(daily_data):,} stocks.")
    return daily_data


def prefilter_candidates(
    daily_data: Dict[str, pd.DataFrame],
    backtest_days: int = 189,
    vol_mult_min: float = 1.1,
    ema_vol30_min: float = 10000.0,
    pct_gain_min: float = 1.0,
    close_min: float = 1.0,
    turnover_min: float = 1_000_000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Computes daily technical indicators using a wide capture floor ($1M turnover,
    1% gain, 1.1x volume, 10k 30-day EMA vol, $1 close) so you can interactively
    adjust or pause filters directly in the web dashboard without re-fetching.
    """
    print("[3/5] Evaluating daily volume, % gain, and liquidity rules...")
    candidates: Dict[str, pd.DataFrame] = {}

    for sym, df in daily_data.items():
        df = df.sort_index().copy()
        df["ema_vol_20"] = df["Volume"].ewm(span=20, adjust=False).mean()
        df["ema_vol_30"] = df["Volume"].ewm(span=30, adjust=False).mean()
        df["sma_vol_20"] = df["Volume"].rolling(20).mean()
        df["pct_chg"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100.0
        df["vol_mult"] = df["Volume"] / df["ema_vol_20"]

        # Typical price proxy before 5-minute intraday VWAP refinement
        df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["ema_vwap_20"] = df["vwap"].ewm(span=20, adjust=False).mean()
        df["turnover_day"] = df["vwap"] * df["Volume"]
        df["turnover_20d"] = df["ema_vwap_20"] * df["sma_vol_20"]

        window = df.iloc[-backtest_days:].copy()
        cond = (
            (window["vol_mult"] >= vol_mult_min)
            & (window["ema_vol_30"] >= ema_vol30_min)
            & (window["pct_chg"] >= pct_gain_min)
            & (window["Close"] > close_min)
            & ((window["turnover_day"] >= turnover_min) | (window["turnover_20d"] >= turnover_min))
        )

        if cond.any():
            window["pre_pass"] = cond
            candidates[sym] = window

    print(f"      {len(candidates):,} candidate tickers passed technical/liquidity pre-filter.")
    return candidates


def fetch_fundamentals_and_true_intraday_vwap(
    candidates: Dict[str, pd.DataFrame],
    min_mkt_cap_m: float = 150.0,
) -> Tuple[Dict[str, dict], Dict[str, Dict[str, float]]]:
    """
    1. Fetches Market Cap (> $150M capture floor) and GICS Sector for candidates.
    2. Downloads 5-minute intraday bars (last 60 trading days) for qualifying
       stocks to compute TRUE intraday VWAP = sum(TypicalPrice * Vol) / sum(Vol).
    """
    print(f"[4/5] Checking Market Cap > ${min_mkt_cap_m:.0f}M & Sector metadata...")
    qualified_meta: Dict[str, dict] = {}
    syms = list(candidates.keys())

    for idx, sym in enumerate(syms, start=1):
        if idx % 100 == 0:
            print(f"      Checked metadata for {idx}/{len(syms)} candidates...")
        try:
            tk = yf.Ticker(sym)
            finfo = tk.fast_info
            mkt_cap = getattr(finfo, "market_cap", None)
            if mkt_cap is None and isinstance(finfo, dict):
                mkt_cap = finfo.get("market_cap")
            if not mkt_cap or (mkt_cap / 1_000_000.0) <= min_mkt_cap_m:
                continue

            info = tk.info or {}
            raw_sector = info.get("sector", "Technology")
            mapped_sector = SECTOR_MAP.get(raw_sector, "technology")

            qualified_meta[sym] = {
                "mktCapM": round(mkt_cap / 1_000_000.0, 1),
                "sector": mapped_sector,
            }
        except Exception:
            continue

    print(f"      {len(qualified_meta):,} stocks passed Market Cap > ${min_mkt_cap_m:.0f}M.")

    # Compute TRUE Intraday VWAP from 5-minute bars (available for last 60 trading days)
    qual_syms = list(qualified_meta.keys())
    true_vwap_map: Dict[str, Dict[str, float]] = {s: {} for s in qual_syms}
    if not qual_syms:
        return qualified_meta, true_vwap_map

    print("[5/5] Computing TRUE Intraday VWAP from 5-minute bars (last 60 trading days)...")
    batch_size = 100
    for b_idx in range(math.ceil(len(qual_syms) / batch_size)):
        chunk = qual_syms[b_idx * batch_size : (b_idx + 1) * batch_size]
        try:
            intra = yf.download(
                tickers=chunk,
                period="60d",
                interval="5m",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if intra.empty:
            continue

        is_multi = isinstance(intra.columns, pd.MultiIndex)
        for sym in chunk:
            try:
                if is_multi:
                    if sym in intra.columns.get_level_values(0):
                        idf = intra[sym].copy()
                    elif sym in intra.columns.get_level_values(1):
                        idf = intra.xs(sym, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    idf = intra.copy()
                idf = idf.dropna(subset=["High", "Low", "Close", "Volume"])
                idf = idf[idf["Volume"] > 0]
                if idf.empty:
                    continue

                idf["tp"] = (idf["High"] + idf["Low"] + idf["Close"]) / 3.0
                idf["pv"] = idf["tp"] * idf["Volume"]
                dt_idx = pd.to_datetime(idf.index)
                if getattr(dt_idx, "tz", None) is not None:
                    dt_idx = dt_idx.tz_convert("America/New_York").tz_localize(None)
                idf["trade_date"] = dt_idx.strftime("%Y-%m-%d")

                grouped = idf.groupby("trade_date", as_index=True).agg(
                    pv_sum=("pv", "sum"),
                    vol_sum=("Volume", "sum"),
                )
                grouped = grouped[grouped["vol_sum"] > 0]
                grouped["true_vwap"] = grouped["pv_sum"] / grouped["vol_sum"]

                for d_str, v_val in grouped["true_vwap"].items():
                    true_vwap_map[sym][str(d_str)] = round(float(v_val), 4)
            except Exception:
                continue

        time.sleep(0.4)

    return qualified_meta, true_vwap_map


def build_json_payload(
    universe: pd.DataFrame,
    candidates: Dict[str, pd.DataFrame],
    qualified_meta: Dict[str, dict],
    true_vwap_map: Dict[str, Dict[str, float]],
    turnover_min: float = 1_000_000.0,
    capture_floor: Optional[dict] = None,
) -> dict:
    """
    Assembles the final JSON structure consumed by us_backtest_screener.html.
    """
    name_lookup = dict(zip(universe["Symbol"], universe["CleanName"]))
    exch_lookup = dict(zip(universe["Symbol"], universe["Exchange"]))

    all_dates_set = set()
    for sym in qualified_meta:
        df = candidates[sym]
        for dt in df.index:
            all_dates_set.add(pd.Timestamp(dt).strftime("%Y-%m-%d"))

    sorted_dates = sorted(all_dates_set)
    history_by_date: Dict[str, List[dict]] = {d: [] for d in sorted_dates}

    for sym, meta in qualified_meta.items():
        df = candidates[sym].copy()
        date_strs = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in df.index]
        df["date_str"] = date_strs

        # Replace proxy VWAP with true intraday 5m VWAP wherever available
        df["final_vwap"] = [
            true_vwap_map.get(sym, {}).get(d_str, float(proxy_v))
            for d_str, proxy_v in zip(df["date_str"], df["vwap"])
        ]
        df["final_ema_vwap_20"] = df["final_vwap"].ewm(span=20, adjust=False).mean()
        df["final_turnover_day"] = df["final_vwap"] * df["Volume"]
        df["final_turnover_20d"] = df["final_ema_vwap_20"] * df["sma_vol_20"]

        # Re-verify exact Chartink VWAP turnover rule using true intraday VWAP
        final_pass = df["pre_pass"] & (
            (df["final_turnover_day"] >= turnover_min) | (df["final_turnover_20d"] >= turnover_min)
        )

        passed_rows = df[final_pass]
        for _, row in passed_rows.iterrows():
            d_str = row["date_str"]
            history_by_date[d_str].append(
                {
                    "symbol": sym,
                    "name": name_lookup.get(sym, sym),
                    "exchange": exch_lookup.get(sym, "NASDAQ"),
                    "sector": meta["sector"],
                    "price": round(float(row["Close"]), 2),
                    "intradayVwap": round(float(row["final_vwap"]), 2),
                    "pctChange": round(float(row["pct_chg"]), 2),
                    "volume": int(row["Volume"]),
                    "emaVol30": int(row["ema_vol_30"]),
                    "volMult": round(float(row["vol_mult"]), 2),
                    "mktCapM": meta["mktCapM"],
                    "turnover": int(row["final_turnover_day"]),
                    "turnover20d": int(row["final_turnover_20d"]),
                }
            )

    history_list = []
    for d_str in sorted_dates:
        stocks_sorted = sorted(history_by_date[d_str], key=lambda x: x["pctChange"], reverse=True)
        history_list.append({"date": d_str, "stocks": stocks_sorted})

    # Include last 60 days of real OHLCV candles for latest triggered stocks
    latest_stocks = history_list[-1]["stocks"] if history_list else []
    candles_dict: Dict[str, List[dict]] = {}
    for st in latest_stocks[:60]:
        sym = st["symbol"]
        if sym in candidates:
            sub = candidates[sym].tail(60)
            c_list = []
            for dt, r in sub.iterrows():
                d_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                v_val = true_vwap_map.get(sym, {}).get(d_str, float(r["vwap"]))
                c_list.append(
                    {
                        "date": d_str,
                        "o": round(float(r["Open"]), 2),
                        "h": round(float(r["High"]), 2),
                        "l": round(float(r["Low"]), 2),
                        "c": round(float(r["Close"]), 2),
                        "v": int(r["Volume"]),
                        "vwap": round(float(v_val), 2),
                    }
                )
            candles_dict[sym] = c_list

    latest_date_str = history_list[-1]["date"] if history_list else datetime.now().strftime("%Y-%m-%d")
    floor_meta = capture_floor or {
        "minTurnoverM": round(turnover_min / 1_000_000.0, 2),
        "minGain": 1.0,
        "volMult": 1.1,
        "minEmaVol": 10000.0,
        "minPrice": 1.0,
        "minMktCapM": 150.0,
    }
    return {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latestTradingDate": latest_date_str,
        "source": "Yahoo Finance EOD + 5m Intraday VWAP",
        "captureFloor": floor_meta,
        "history": history_list,
        "candles": candles_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch US EOD + True Intraday VWAP Screener JSON")
    parser.add_argument("--output", default="us_eod_data.json", help="Output JSON path (default: us_eod_data.json)")
    parser.add_argument("--days", type=int, default=189, help="Backtest history trading days (default: 189 = 9 months)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional symbol limit for quick testing (0 = all)")
    parser.add_argument("--min-turnover", type=float, default=1.0, help="Min VWAP turnover in $M (default: 1.0)")
    parser.add_argument("--min-gain", type=float, default=1.0, help="Min daily %% gain (default: 1.0)")
    parser.add_argument("--min-vol-mult", type=float, default=1.1, help="Min volume multiplier vs 20d EMA (default: 1.1)")
    parser.add_argument("--min-ema-vol", type=float, default=10000.0, help="Min 30-day EMA volume (default: 10000)")
    parser.add_argument("--min-price", type=float, default=1.0, help="Min closing price in USD (default: 1.0)")
    parser.add_argument("--min-mkt-cap", type=float, default=150.0, help="Min market cap in $M (default: 150.0)")
    args = parser.parse_args()

    turnover_dollars = args.min_turnover * 1_000_000.0
    capture_floor = {
        "minTurnoverM": args.min_turnover,
        "minGain": args.min_gain,
        "volMult": args.min_vol_mult,
        "minEmaVol": args.min_ema_vol,
        "minPrice": args.min_price,
        "minMktCapM": args.min_mkt_cap,
    }

    universe = fetch_us_common_stocks(max_symbols=args.max_symbols)
    daily_data = download_daily_ohlcv_batched(universe["Symbol"].tolist(), batch_size=250)
    candidates = prefilter_candidates(
        daily_data,
        backtest_days=args.days,
        vol_mult_min=args.min_vol_mult,
        ema_vol30_min=args.min_ema_vol,
        pct_gain_min=args.min_gain,
        close_min=args.min_price,
        turnover_min=turnover_dollars,
    )
    qualified_meta, true_vwap_map = fetch_fundamentals_and_true_intraday_vwap(
        candidates,
        min_mkt_cap_m=args.min_mkt_cap,
    )
    payload = build_json_payload(
        universe,
        candidates,
        qualified_meta,
        true_vwap_map,
        turnover_min=turnover_dollars,
        capture_floor=capture_floor,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    latest_count = len(payload["history"][-1]["stocks"]) if payload["history"] else 0
    print("\n========================================================")
    print(f"SUCCESS! Saved real US EOD backtest data to: {args.output}")
    print(f"Latest Session Date : {payload['latestTradingDate']}")
    print(f"Latest Matches      : {latest_count} stocks")
    print(f"Total Backtest Days : {len(payload['history'])} trading days")
    print("Open us_backtest_screener.html and click 'Load Latest EOD'!")
    print("========================================================")


if __name__ == "__main__":
    main()
