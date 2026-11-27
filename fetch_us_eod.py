"""
US Market EOD & True Intraday VWAP Screener Data Generator
==========================================================
Fetches All US Equities (NYSE, NASDAQ, NYSE American/AMEX, CBOE, OTC & ADRs),
evaluates your breakout scan across the last 9 months (189 trading days),
computes TRUE intraday VWAP from 5-minute bars, and exports `us_eod_data.json`
for the Chartink-style US dashboard.

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
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None


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


class YahooCrumbSession:
    """
    Maintains an authenticated Yahoo Finance cookie + crumb session with
    automatic refresh and exponential backoff on HTTP 401 (Invalid Crumb)
    and HTTP 429 (Too Many Requests).
    """

    BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }

    def __init__(self) -> None:
        self.session = self._new_session()
        self.crumb: Optional[str] = None

    def _new_session(self):
        if cffi_requests is not None:
            return cffi_requests.Session(impersonate="chrome")
        sess = requests.Session()
        sess.headers.update(self.BROWSER_HEADERS)
        return sess

    def refresh(self, max_attempts: int = 3) -> bool:
        """Acquires fresh Yahoo consent cookies and a valid API crumb."""
        for attempt in range(max_attempts):
            try:
                self.session = self._new_session()
                self.crumb = None

                # Step 1: Seed session cookies from Yahoo consent/finance endpoints
                for seed_url in ("https://fc.yahoo.com", "https://finance.yahoo.com/quote/AAPL"):
                    try:
                        self.session.get(seed_url, timeout=15, allow_redirects=True)
                    except Exception:
                        pass

                # Step 2: Request fresh crumb tied to the seeded cookies
                for crumb_url in (
                    "https://query1.finance.yahoo.com/v1/test/getcrumb",
                    "https://query2.finance.yahoo.com/v1/test/getcrumb",
                ):
                    resp = self.session.get(crumb_url, timeout=15)
                    text = (resp.text or "").strip()
                    if (
                        resp.status_code == 200
                        and text
                        and "<html" not in text.lower()
                        and "too many requests" not in text.lower()
                    ):
                        self.crumb = text
                        return True
            except Exception:
                pass

            backoff = 1.5 * (2 ** attempt) + random.uniform(0.2, 0.8)
            time.sleep(backoff)

        return False

    def get_json(self, url: str, params: Optional[dict] = None, max_retries: int = 4) -> Optional[dict]:
        """Executes a GET request with crumb injection, 401 crumb refresh, and 429 retry."""
        if not self.crumb:
            self.refresh()

        for attempt in range(max_retries):
            req_params = dict(params or {})
            if self.crumb:
                req_params["crumb"] = self.crumb

            try:
                resp = self.session.get(url, params=req_params, timeout=20)
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (401, 403):
                    print(f"      Yahoo HTTP {resp.status_code} (Invalid Crumb) — refreshing session (retry {attempt + 1}/{max_retries})...")
                    time.sleep(1.5 * (2 ** attempt))
                    self.refresh()
                    continue

                if resp.status_code == 429:
                    wait_s = 3.0 * (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"      Yahoo HTTP 429 rate limit — waiting {wait_s:.1f}s (retry {attempt + 1}/{max_retries})...")
                    time.sleep(wait_s)
                    self.refresh()
                    continue
            except Exception:
                time.sleep(1.5 * (2 ** attempt))

        return None


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts numeric/pandas values to int, returning default on NaN/Inf/None."""
    if val is None:
        return default
    try:
        fval = float(val)
        if not math.isfinite(fval):
            return default
        return int(round(fval))
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, digits: int = 2, default: float = 0.0) -> float:
    """Safely converts numeric/pandas values to rounded float, returning default on NaN/Inf/None."""
    if val is None:
        return default
    try:
        fval = float(val)
        if not math.isfinite(fval):
            return default
        return round(fval, digits)
    except (TypeError, ValueError):
        return default


def _download_yf_with_retry(
    tickers: List[str],
    period: str,
    interval: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Wraps yf.download with exponential backoff retries against transient 401/429 responses."""
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            print(f"      Warning: yf.download attempt {attempt + 1}/{max_retries} failed: {exc}")

        if attempt < max_retries - 1:
            wait_s = 2.5 * (2 ** attempt) + random.uniform(0.3, 1.0)
            time.sleep(wait_s)

    return pd.DataFrame()


def _clean_company_name(raw_name: str) -> str:
    """Strips boilerplate security suffixes from exchange directory names."""
    name = str(raw_name or "").strip()
    for suffix in (
        " - Common Stock",
        " Common Stock",
        " - Ordinary Shares",
        " Ordinary Shares",
        " - Class A Common Stock",
        " Class A Common Stock",
        " - American Depositary Shares",
    ):
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)].strip()
    return name or "Unknown"


def _is_valid_common_ticker(sym: str) -> bool:
    """Filters out warrants, rights, units, preferreds, and test symbols."""
    if not sym or len(sym) > 5:
        return False
    if any(ch in sym for ch in ("$", "^", "/", " ", "~", "#")):
        return False
    clean = sym.replace(".", "-")
    parts = clean.split("-")
    if len(parts) > 2 or not parts[0].isalpha():
        return False
    if len(parts) == 2 and parts[1] not in ("A", "B"):
        return False
    # 5-letter NASDAQ tickers ending in W/R/U/P/Q denote warrants, rights, units, preferred, or bankruptcy
    if len(parts[0]) == 5 and parts[0][-1] in ("W", "R", "U", "P", "Q"):
        return False
    return True


def fetch_us_common_stocks(max_symbols: int = 0) -> pd.DataFrame:
    """
    Fetches the active All-US equity universe (NYSE, NASDAQ, NYSE American/AMEX,
    CBOE, OTC & ADRs) using the official SEC EDGAR exchange directory first
    (cloud-IP friendly), with NASDAQ Trader symbol files as automatic fallback.
    """
    print("[1/5] Fetching All-US equity universe (NYSE, NASDAQ, AMEX, CBOE, OTC & ADRs)...")
    records: List[dict] = []

    exchange_normalize = {
        "NASDAQ": "NASDAQ",
        "NYSE": "NYSE",
        "NYSE AMERICAN": "AMEX",
        "AMEX": "AMEX",
        "NYSE MKT": "AMEX",
        "NYSE ARCA": "NYSE",
        "CBOE": "CBOE",
        "BATS": "CBOE",
        "CBOE BZX": "CBOE",
        "CBOE BYX": "CBOE",
        "CBOE EDGX": "CBOE",
        "CBOE EDGA": "CBOE",
        "OTC": "OTC",
        "OTCBB": "OTC",
        "OTCQX": "OTC",
        "OTCQB": "OTC",
        "PINK": "OTC",
        "": "OTC",
        "NONE": "OTC",
    }

    # Primary source: Official SEC EDGAR company_tickers_exchange.json
    try:
        sec_headers = {
            "User-Agent": "KokotradesUSMarketScreener/1.0 (contact@kokotrades.github.io)",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers_exchange.json",
            headers=sec_headers,
            timeout=25,
        )
        if resp.status_code == 200:
            data = resp.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])
            if "ticker" in fields and "name" in fields and "exchange" in fields:
                idx_ticker = fields.index("ticker")
                idx_name = fields.index("name")
                idx_exch = fields.index("exchange")

                for row in rows:
                    exch_raw = str(row[idx_exch] or "").strip().upper()
                    mapped_exch = exchange_normalize.get(exch_raw)
                    if not mapped_exch:
                        continue

                    sym = str(row[idx_ticker] or "").strip().upper().replace(".", "-")
                    if not _is_valid_common_ticker(sym):
                        continue

                    records.append(
                        {
                            "Symbol": sym,
                            "CleanName": _clean_company_name(row[idx_name]),
                            "Exchange": mapped_exch,
                        }
                    )
    except Exception as exc:
        print(f"      SEC EDGAR directory notice: {exc} — trying NASDAQ Trader fallback...")

    # Fallback source: NASDAQ Trader symbol directories (NASDAQ, NYSE, AMEX, CBOE)
    if not records:
        other_exch_map = {
            "N": "NYSE",
            "A": "AMEX",
            "P": "NYSE",
            "Z": "CBOE",
            "V": "NYSE",
        }
        browser_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
        }
        for url, is_nasdaq in (
            ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", True),
            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", False),
        ):
            try:
                r = requests.get(url, headers=browser_headers, timeout=25)
                if r.status_code != 200:
                    continue
                df_txt = pd.read_csv(io.StringIO(r.text), sep="|")
                sym_col = "Symbol" if is_nasdaq else "ACT Symbol"
                if sym_col not in df_txt.columns:
                    continue

                df_txt = df_txt[df_txt["Test Issue"].astype(str).str.upper() == "N"]
                if "ETF" in df_txt.columns:
                    df_txt = df_txt[df_txt["ETF"].astype(str).str.upper() == "N"]

                for _, row in df_txt.iterrows():
                    sym = str(row[sym_col] or "").strip().upper().replace(".", "-")
                    if not _is_valid_common_ticker(sym):
                        continue
                    if is_nasdaq:
                        mapped_exch = "NASDAQ"
                    else:
                        ex_code = str(row.get("Exchange", "") or "").strip().upper()
                        mapped_exch = other_exch_map.get(ex_code)
                        if not mapped_exch:
                            continue
                    records.append(
                        {
                            "Symbol": sym,
                            "CleanName": _clean_company_name(row.get("Security Name", sym)),
                            "Exchange": mapped_exch,
                        }
                    )
            except Exception as exc:
                print(f"      Warning fetching {url}: {exc}")

    if not records:
        raise RuntimeError("Unable to retrieve US symbol list from SEC EDGAR or NASDAQ Trader.")

    universe = pd.DataFrame(records).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    if max_symbols and max_symbols > 0:
        universe = universe.head(max_symbols).copy()

    exch_counts = universe["Exchange"].value_counts().to_dict()
    print(f"      Loaded {len(universe):,} US stocks & ADRs ({exch_counts}).")
    return universe


def download_daily_ohlcv_batched(symbols: List[str], batch_size: int = 200) -> Dict[str, pd.DataFrame]:
    """
    Downloads 3 years of daily OHLCV bars in batches (providing full 14-month
    warmup for Weekly & Monthly Wilder RSI across the entire 9-month backtest).
    """
    print(f"[2/5] Downloading 3-year daily OHLCV bars for {len(symbols):,} symbols...")
    daily_data: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return daily_data

    total_batches = math.ceil(len(symbols) / batch_size)
    for b_idx in range(total_batches):
        chunk = symbols[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"      Batch {b_idx + 1}/{total_batches} ({len(chunk)} symbols)...")
        raw = _download_yf_with_retry(chunk, period="3y", interval="1d", max_retries=3)
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
                if len(df) >= 60:
                    daily_data[sym] = df
            except Exception:
                continue

        time.sleep(0.75)

    print(f"      Loaded valid daily history for {len(daily_data):,} stocks.")
    return daily_data


def _compute_point_in_time_mtf_rsi(close_series: pd.Series, freq: str, period: int = 14) -> pd.Series:
    """
    Computes point-in-time Wilder RSI(period) on a higher timeframe ('W-FRI' or 'M')
    for every daily bar without look-ahead bias. On mid-week or mid-month days,
    uses completed prior periods' Wilder average gain/loss plus the current
    in-progress period's close.
    """
    idx = pd.DatetimeIndex(close_series.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)

    closes = close_series.to_numpy(dtype=float)
    n = len(closes)
    if n == 0:
        return pd.Series([], index=close_series.index, dtype=float)

    periods = idx.to_period(freq)
    period_end_positions: List[int] = []
    for i in range(n):
        if i == n - 1 or periods[i] != periods[i + 1]:
            period_end_positions.append(i)

    num_periods = len(period_end_positions)
    if num_periods < 2:
        return pd.Series(np.full(n, 50.0), index=close_series.index, dtype=float)

    period_closes = np.array([closes[pos] for pos in period_end_positions], dtype=float)
    deltas = np.diff(period_closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    alpha = 1.0 / float(period)
    avg_gain_after = np.zeros(num_periods, dtype=float)
    avg_loss_after = np.zeros(num_periods, dtype=float)

    avg_gain_after[1] = gains[0]
    avg_loss_after[1] = losses[0]
    for p in range(2, num_periods):
        avg_gain_after[p] = (1.0 - alpha) * avg_gain_after[p - 1] + alpha * gains[p - 1]
        avg_loss_after[p] = (1.0 - alpha) * avg_loss_after[p - 1] + alpha * losses[p - 1]

    rsi_out = np.full(n, 50.0, dtype=float)
    p_idx = 0
    for i in range(n):
        while p_idx < num_periods - 1 and i > period_end_positions[p_idx]:
            p_idx += 1

        if p_idx == 0:
            rsi_out[i] = 50.0
            continue

        prev_close = period_closes[p_idx - 1]
        cur_delta = closes[i] - prev_close
        cur_gain = cur_delta if cur_delta > 0.0 else 0.0
        cur_loss = -cur_delta if cur_delta < 0.0 else 0.0

        if p_idx == 1:
            ag = cur_gain
            al = cur_loss
        else:
            ag = (1.0 - alpha) * avg_gain_after[p_idx - 1] + alpha * cur_gain
            al = (1.0 - alpha) * avg_loss_after[p_idx - 1] + alpha * cur_loss

        if al == 0.0:
            rsi_out[i] = 100.0 if ag > 0.0 else 50.0
        else:
            rs = ag / al
            rsi_out[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi_out, index=close_series.index, dtype=float)


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
    Computes daily technical indicators plus point-in-time Weekly RSI(14) and
    Monthly RSI(14) using a wide capture floor so you can interactively adjust
    or pause filters directly in the web dashboard without re-fetching.
    """
    print("[3/5] Evaluating daily volume, % gain, liquidity, and 1W/1M RSI(14)...")
    candidates: Dict[str, pd.DataFrame] = {}

    for sym, df in daily_data.items():
        df = df.sort_index().copy()
        df["ema_vol_20"] = df["Volume"].ewm(span=20, adjust=False).mean()
        df["ema_vol_30"] = df["Volume"].ewm(span=30, adjust=False).mean()
        df["sma_vol_20"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["pct_chg"] = ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100.0).fillna(0.0)
        df["vol_mult"] = (df["Volume"] / df["ema_vol_20"].replace(0, np.nan)).fillna(0.0)
        df["rsi_14_w"] = _compute_point_in_time_mtf_rsi(df["Close"], freq="W-FRI", period=14)
        df["rsi_14_m"] = _compute_point_in_time_mtf_rsi(df["Close"], freq="M", period=14)

        # Typical price proxy before 5-minute intraday VWAP refinement
        df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["ema_vwap_20"] = df["vwap"].ewm(span=20, adjust=False).mean()
        df["turnover_day"] = (df["vwap"] * df["Volume"]).fillna(0.0)
        df["turnover_20d"] = (df["ema_vwap_20"] * df["sma_vol_20"]).fillna(df["turnover_day"])

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
    1. Fetches Market Cap (> $150M capture floor) and GICS Sector via authenticated
       Yahoo batch quote requests (with automatic crumb refresh & retry), avoiding
       per-ticker quoteSummary HTTP 401 crumb failures.
    2. Downloads 5-minute intraday bars (last 60 trading days) for qualifying
       stocks to compute TRUE intraday VWAP = sum(TypicalPrice * Vol) / sum(Vol).
    """
    print(f"[4/5] Checking Market Cap > ${min_mkt_cap_m:.0f}M & Sector metadata...")
    qualified_meta: Dict[str, dict] = {}
    syms = list(candidates.keys())
    if not syms:
        return qualified_meta, {}

    yahoo_auth = YahooCrumbSession()
    crumb_ok = yahoo_auth.refresh()
    if crumb_ok:
        print("      Authenticated Yahoo Finance session & crumb acquired.")
    else:
        print("      Warning: Could not pre-fetch Yahoo crumb; will retry during batch requests.")

    # Batch fetch marketCap + sector in chunks of 100 symbols (100x fewer API calls)
    meta_batch_size = 100
    unresolved_syms: List[str] = []

    for b_idx in range(math.ceil(len(syms) / meta_batch_size)):
        chunk = syms[b_idx * meta_batch_size : (b_idx + 1) * meta_batch_size]
        payload = yahoo_auth.get_json(
            "https://query2.finance.yahoo.com/v7/finance/quote",
            params={
                "symbols": ",".join(chunk),
                "fields": "symbol,marketCap,sector,industry",
            },
            max_retries=4,
        )

        results = (payload or {}).get("quoteResponse", {}).get("result", [])
        if not results:
            unresolved_syms.extend(chunk)
            time.sleep(0.6)
            continue

        seen_in_batch = set()
        for item in results:
            sym = item.get("symbol")
            if not sym:
                continue
            seen_in_batch.add(sym)
            mkt_cap = item.get("marketCap")
            if not mkt_cap or (float(mkt_cap) / 1_000_000.0) <= min_mkt_cap_m:
                continue

            raw_sector = item.get("sector") or "Technology"
            mapped_sector = SECTOR_MAP.get(raw_sector, "technology")
            qualified_meta[sym] = {
                "mktCapM": _safe_float(float(mkt_cap) / 1_000_000.0, digits=1, default=0.0),
                "sector": mapped_sector,
            }

        for sym in chunk:
            if sym not in seen_in_batch:
                unresolved_syms.append(sym)

        time.sleep(0.5)

    # Fallback via fast_info for any symbols not returned by batch quote
    if unresolved_syms:
        print(f"      Checking fast_info fallback for {len(unresolved_syms)} remaining symbols...")
        for sym in unresolved_syms:
            try:
                tk = yf.Ticker(sym)
                finfo = tk.fast_info
                mkt_cap = getattr(finfo, "market_cap", None)
                if mkt_cap is None and isinstance(finfo, dict):
                    mkt_cap = finfo.get("market_cap")
                if not mkt_cap or (float(mkt_cap) / 1_000_000.0) <= min_mkt_cap_m:
                    continue

                qualified_meta[sym] = {
                    "mktCapM": _safe_float(float(mkt_cap) / 1_000_000.0, digits=1, default=0.0),
                    "sector": "technology",
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
    batch_size = 75
    for b_idx in range(math.ceil(len(qual_syms) / batch_size)):
        chunk = qual_syms[b_idx * batch_size : (b_idx + 1) * batch_size]
        intra = _download_yf_with_retry(chunk, period="60d", interval="5m", max_retries=3)

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
                    if pd.notna(v_val) and math.isfinite(float(v_val)):
                        true_vwap_map[sym][str(d_str)] = round(float(v_val), 4)
            except Exception:
                continue

        time.sleep(0.75)

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
    Assembles the final JSON structure consumed by us_backtest_screener.html,
    safely sanitizing any NaN/Inf turnover or indicator values before serialization.
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
            true_vwap_map.get(sym, {}).get(d_str, _safe_float(proxy_v, digits=4, default=0.0))
            for d_str, proxy_v in zip(df["date_str"], df["vwap"])
        ]
        df["final_vwap"] = pd.to_numeric(df["final_vwap"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["final_vwap"] = df["final_vwap"].fillna(df["Close"]).fillna(0.0)

        df["final_ema_vwap_20"] = df["final_vwap"].ewm(span=20, adjust=False).mean().fillna(df["final_vwap"])
        sma_vol_safe = df["sma_vol_20"].fillna(df["Volume"]).fillna(0.0)

        df["final_turnover_day"] = (
            (df["final_vwap"] * df["Volume"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        df["final_turnover_20d"] = (
            (df["final_ema_vwap_20"] * sma_vol_safe)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(df["final_turnover_day"])
            .fillna(0.0)
        )

        # Re-verify exact Chartink VWAP turnover rule using true intraday VWAP
        final_pass = (
            df["pre_pass"].fillna(False)
            & df["Close"].notna()
            & (df["Close"] > 0)
            & df["Volume"].notna()
            & (df["Volume"] > 0)
            & ((df["final_turnover_day"] >= turnover_min) | (df["final_turnover_20d"] >= turnover_min))
        )

        passed_rows = df[final_pass]
        for _, row in passed_rows.iterrows():
            d_str = row["date_str"]
            day_turnover = _safe_int(row["final_turnover_day"], default=0)
            avg_turnover_20d = _safe_int(row["final_turnover_20d"], default=day_turnover)

            history_by_date[d_str].append(
                {
                    "symbol": sym,
                    "name": name_lookup.get(sym, sym),
                    "exchange": exch_lookup.get(sym, "NASDAQ"),
                    "sector": meta["sector"],
                    "price": _safe_float(row["Close"], digits=2, default=0.0),
                    "intradayVwap": _safe_float(row["final_vwap"], digits=2, default=0.0),
                    "pctChange": _safe_float(row["pct_chg"], digits=2, default=0.0),
                    "volume": _safe_int(row["Volume"], default=0),
                    "emaVol30": _safe_int(row["ema_vol_30"], default=0),
                    "volMult": _safe_float(row["vol_mult"], digits=2, default=0.0),
                    "mktCapM": _safe_float(meta["mktCapM"], digits=1, default=0.0),
                    "turnover": day_turnover,
                    "turnover20d": avg_turnover_20d,
                    "rsi14W": _safe_float(row.get("rsi_14_w", 50.0), digits=1, default=50.0),
                    "rsi14M": _safe_float(row.get("rsi_14_m", 50.0), digits=1, default=50.0),
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
                v_val = true_vwap_map.get(sym, {}).get(d_str, _safe_float(r["vwap"], digits=2, default=0.0))
                c_list.append(
                    {
                        "date": d_str,
                        "o": _safe_float(r["Open"], digits=2, default=0.0),
                        "h": _safe_float(r["High"], digits=2, default=0.0),
                        "l": _safe_float(r["Low"], digits=2, default=0.0),
                        "c": _safe_float(r["Close"], digits=2, default=0.0),
                        "v": _safe_int(r["Volume"], default=0),
                        "vwap": _safe_float(v_val, digits=2, default=0.0),
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
        "universe": "All US (NYSE, NASDAQ, AMEX, CBOE, OTC & ADRs)",
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
    daily_data = download_daily_ohlcv_batched(universe["Symbol"].tolist(), batch_size=200)
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
