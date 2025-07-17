#!/usr/bin/env python3
# Financial Fetcher – robust edition (rev-H2, 13 Jun 2025)
# ─────────────────────────────────────────────────────────
#  ✧ Calendar-agnostic fiscal-quarter labelling (new in H2)
#  ✧ Everything else identical to rev-H
# ─────────────────────────────────────────────────────────
import subprocess, sys, os, warnings, re, requests, time, json, calendar
from difflib import SequenceMatcher
from datetime import datetime, timedelta

# ──────────── auto-install ───────────────────────────────────────────────
_req = [
    ("pandas",           "pandas"),
    ("openpyxl",         "openpyxl"),
    ("requests",         "requests"),
    ("yfinance>=0.2.62", "yfinance"),
]
for pkg, mod in _req:
    try:
        __import__(mod.split(">=")[0])
    except ImportError:
        print(f"Installing {pkg}…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pkg, "--no-cache-dir"]
        )

# ──────────── imports ────────────────────────────────────────────────────
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────── name → candidates ─────────────────────────────────────
_candidate_cache: dict[str, list[str]] = {}

def resolve_ticker_candidates(name: str) -> list[str]:
    if name in _candidate_cache:
        return _candidate_cache[name]

    url    = "https://query1.finance.yahoo.com/v1/finance/search"
    hdr    = {"User-Agent": "Mozilla/5.0"}
    params = {"q": name, "quotesCount": 20, "newsCount": 0}
    try:
        quotes = requests.get(url, headers=hdr, params=params,
                              timeout=5).json().get("quotes", [])
    except Exception:
        quotes = []

    def score(q):
        cand = q.get("shortname") or q.get("longname") or ""
        return SequenceMatcher(None, name.lower(), cand.lower()).ratio()

    syms = [q["symbol"] for q in sorted(quotes, key=score, reverse=True)
            if q.get("symbol")]
    _candidate_cache[name] = syms
    return syms

# ───────────────── price helper ──────────────────────────────────────────
def build_price_lookup(tkr: yf.Ticker, start, end):
    try:
        closes = (
            tkr.history(
                start=(start - timedelta(days=7)).strftime("%Y-%m-%d"),
                end  =(end   + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
            )["Close"]
            .copy()
        )
        if closes.empty:
            return lambda _dt: (None, None)
        closes.index = closes.index.tz_localize(None)

        def price(dt: pd.Timestamp):
            le = closes[closes.index <= dt]
            if not le.empty:
                return le.iloc[-1], le.index[-1]
            gt = closes[closes.index > dt]
            return (gt.iloc[0], gt.index[0]) if not gt.empty else (None, None)

        return price
    except Exception:
        return lambda _dt: (None, None)

# ───────────── CLI ───────────────────────────────────────────────────────
print("\n=== Financial Fetcher (rev-H2) ===\n")
path = input("1) Excel file (.xlsx):\n> ").strip('"')
sh   = input("2) Sheet name containing 'Company Name':\n> ").strip('"')
y0   = input("3) Starting fiscal year (YYYY):\n> ").strip()
if not (y0.isdigit() and len(y0) == 4):
    sys.exit("Invalid year");  y0 = int(y0)
y0 = int(y0)

if not os.path.isfile(path):
    sys.exit(f"File not found → {path}")
try:
    df_cmp = pd.read_excel(path, sheet_name=sh, engine="openpyxl")
except Exception as e:
    sys.exit(f"Cannot read sheet → {e}")
if "Company Name" not in df_cmp.columns:
    sys.exit("'Company Name' column missing")

companies = df_cmp["Company Name"].astype(str).unique()
print(f"Found {len(companies)} companies.\n")

# ─────────── label dictionaries ──────────────────────────────────────────
BAL_KEEP = [
    "CashAndCashEquivalents","Cash",
    "CashCashEquivalentsAndShortTermInvestments",
    "CashAndCashEquivalentsAtCarryingValue",
    "ShortTermInvestments",
    "TotalDebt","LongTermDebt","DebtNoncurrent",
    "ShortLongTermDebt","ShortTermDebt",
    "CurrentPortionOfLongTermDebt","DebtCurrent",
    "PreferredStockEquity",
]
BAL_MAP = {
    "CashAndCashEquivalents":"Cash","Cash":"Cash",
    "CashCashEquivalentsAndShortTermInvestments":"Cash",
    "CashAndCashEquivalentsAtCarryingValue":"Cash",
    "ShortTermInvestments":"Cash",
    "TotalDebt":"Total Debt","LongTermDebt":"Long-Term Debt",
    "DebtNoncurrent":"Long-Term Debt","ShortLongTermDebt":"Short-Term Debt",
    "ShortTermDebt":"Short-Term Debt","CurrentPortionOfLongTermDebt":"Short-Term Debt",
    "DebtCurrent":"Short-Term Debt","PreferredStockEquity":"Preferred Stock",
}
INC_KEEP = [
    "TotalRevenue","Revenue",
    "SalesRevenueNet","OperatingRevenue","TotalInterestIncome",
]
INC_MAP = {
    "TotalRevenue":"Revenue","Revenue":"Revenue",
    "SalesRevenueNet":"Revenue","OperatingRevenue":"Revenue",
    "TotalInterestIncome":"Revenue",
}

def reshape(stmt: pd.DataFrame, keep, rename):
    if stmt is None or stmt.empty:
        return pd.DataFrame()
    df = stmt.loc[[r for r in keep if r in stmt.index]].rename(index=rename).T
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df.dropna(how="all")

# ───────── fiscal-quarter helper ─────────────────────────────────────────
def fiscal_quarter_labels(dates: pd.Index, fiscal_year_end_month: int) -> dict[pd.Timestamp, str]:
    """Return {date: 'Qn YYYY'} using the given fiscal-year-end month."""
    freq = f"Q-{calendar.month_abbr[fiscal_year_end_month].upper()}"
    labels = {}
    for dt in dates:
        p = dt.to_period(freq)       # e.g. 2025Q1
        labels[dt] = f"Q{p.quarter} {p.year}"
    return labels

# ───────── main loop ─────────────────────────────────────────────────────
rows, unresolved, diagnostics = [], [], []
hdr = {"User-Agent": "Mozilla/5.0"}

for cname in companies:
    # ─── 1. Resolve best ticker ─────────────────────────────────────────
    candidates = resolve_ticker_candidates(cname)
    if not candidates:
        print(f"→ {cname} → unresolved (no search hits)")
        unresolved.append(cname)
        continue

    selected = None
    for sym in candidates:
        try:
            tkr   = yf.Ticker(sym)
            bs_q  = tkr.get_balance_sheet(freq="quarterly")
            inc_q = tkr.get_income_stmt(freq="quarterly")
            if (bs_q is not None and not bs_q.empty) or \
               (inc_q is not None and not inc_q.empty):
                selected = sym
                break
        except Exception:
            continue
    if not selected:
        print(f"→ {cname} → unresolved (no statements)")
        unresolved.append(cname)
        continue

    sym, tkr = selected, yf.Ticker(selected)
    print(f"→ {cname} → {sym} … ", end="", flush=True)

    # ─── 2. Statement fetch/reshape ─────────────────────────────────────
    bs_y  = tkr.get_balance_sheet(freq="yearly")
    inc_y = tkr.get_income_stmt(freq="yearly")
    bs_q  = tkr.get_balance_sheet(freq="quarterly")
    inc_q = tkr.get_income_stmt(freq="quarterly")
    ydf = reshape(bs_y, BAL_KEEP, BAL_MAP)\
            .join(reshape(inc_y, INC_KEEP, INC_MAP), how="outer")
    qdf = reshape(bs_q, BAL_KEEP, BAL_MAP)\
            .join(reshape(inc_q, INC_KEEP, INC_MAP), how="outer")
    ydf, qdf = (df.loc[:, ~df.columns.duplicated()] for df in (ydf, qdf))
    if "Total Debt" not in qdf.columns:
        ltd = qdf.get("Long-Term Debt", pd.Series(0, index=qdf.index))
        std = qdf.get("Short-Term Debt", pd.Series(0, index=qdf.index))
        qdf["Total Debt"] = ltd.fillna(0) + std.fillna(0)

    # ─── 3. Determine fiscal year-end month ─────────────────────────────
    fye_month = None
    try:
        fye_raw = (tkr.info or {}).get("fiscalYearEnd")          # e.g. 930, 1231
        if isinstance(fye_raw, int) and 101 <= fye_raw <= 1231:
            fye_month = fye_raw // 100
    except Exception:
        pass
    if fye_month is None and not ydf.empty:
        fye_month = int(ydf.index.month.value_counts().idxmax())
    if fye_month is None:
        fye_month = 12                                           # safe default: Dec

    # ─── 4. Build fiscal-quarter labels dict ────────────────────────────
    q_labels = fiscal_quarter_labels(qdf.index, fye_month)

    # ─── 5. Price & shares helpers (unchanged) ──────────────────────────
    shares = tkr.fast_info.get("shares")
    if shares is None:
        try:    shares = (tkr.info or {}).get("sharesOutstanding")
        except: shares = None
    if shares is None:
        try:
            sh_full = tkr.get_shares_full("1900-01-01",
                                           datetime.now().strftime("%Y-%m-%d"))
            shares_series = sh_full.iloc[:, 0] if not sh_full.empty else pd.Series(dtype=float)
        except Exception:
            shares_series = pd.Series(dtype=float)
    else:
        shares_series = pd.Series(dtype=float)

    def shares_for(dt):
        if shares is not None:
            return float(shares)
        if not shares_series.empty:
            s = shares_series[shares_series.index <= dt]
            return None if s.empty else float(s.iloc[-1])
        return None

    dates_all = ydf.index.union(qdf.index)
    price_at  = build_price_lookup(tkr, dates_all.min(), dates_all.max())

    # ─── 6. HTML-JSON fallback (unchanged) ──────────────────────────────
    fallback = {}
    try:
        page   = requests.get(f"https://finance.yahoo.com/quote/{sym}?p={sym}",
                              headers=hdr, timeout=5).text
        js_txt = re.search(r'root\.App\.main = (.*?);\n', page).group(1)
        store  = json.loads(js_txt)["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]
        #  annual
        for stmt in store.get("balanceSheetHistory", {})\
                         .get("balanceSheetStatements", []):
            ed = datetime.fromtimestamp(stmt["endDate"]["raw"]).date()
            d  = fallback.setdefault(("yearly", ed), {})
            d["Cash"]       = stmt.get("cashAndCashEquivalents", {}).get("raw")
            d["Total Debt"] = stmt.get("totalDebt", {}).get("raw")
        for stmt in store.get("incomeStatementHistory", {})\
                         .get("incomeStatementHistory", []):
            ed = datetime.fromtimestamp(stmt["endDate"]["raw"]).date()
            d  = fallback.setdefault(("yearly", ed), {})
            d["Revenue"]    = stmt.get("totalRevenue", {}).get("raw")
        #  quarterly
        for stmt in store.get("balanceSheetHistoryQuarterly", {})\
                         .get("balanceSheetStatements", []):
            ed = datetime.fromtimestamp(stmt["endDate"]["raw"]).date()
            d  = fallback.setdefault(("quarterly", ed), {})
            d["Cash"]       = stmt.get("cashAndCashEquivalents", {}).get("raw")
            d["Total Debt"] = stmt.get("totalDebt", {}).get("raw")
        for stmt in store.get("incomeStatementHistoryQuarterly", {})\
                         .get("incomeStatementHistory", []):
            ed = datetime.fromtimestamp(stmt["endDate"]["raw"]).date()
            d  = fallback.setdefault(("quarterly", ed), {})
            d["Revenue"]    = stmt.get("totalRevenue", {}).get("raw")
    except Exception:
        pass

    # ─── 7. Row-builder (unchanged, except label uses q_labels dict) ────
    def push(dt, row, ptype, plabel):
        debt, cash, rev = row.get("Total Debt"), row.get("Cash"), row.get("Revenue")
        fb = fallback.get((ptype, dt.date()), {})
        if pd.isna(debt) and fb.get("Total Debt") is not None: debt = fb["Total Debt"]
        if pd.isna(cash) and fb.get("Cash") is not None:       cash = fb["Cash"]
        if pd.isna(rev)  and fb.get("Revenue") is not None:    rev  = fb["Revenue"]

        price, used = price_at(dt)
        shs  = shares_for(dt)
        mcap = (price * shs) if price is not None and shs is not None else None
        ev   = (mcap + debt - cash) if mcap is not None and \
                 pd.notna(debt) and pd.notna(cash) else None
        evr  = (ev / rev) if ev is not None and pd.notna(rev) and rev else None

        if None in (price, shs) or any(pd.isna(x) for x in (debt, cash, rev)):
            diagnostics.append({
                "Company Name": cname,
                "Date": dt.date(),
                "Missing Price": price is None,
                "Missing Shares": shs is None,
                "Missing Debt": pd.isna(debt),
                "Missing Cash": pd.isna(cash),
                "Missing Revenue": pd.isna(rev),
            })

        rows.append({
            "Company Name": cname,
            "Verified Ticker": sym,
            "period_type": ptype,
            "period": plabel,
            "Date Fetched": used.strftime("%Y-%m-%d") if used else None,
            "(Net) Debt": (debt - cash) if pd.notna(debt) and pd.notna(cash) else None,
            "Cash": cash,
            "Revenue": rev,
            "Market Cap": mcap,
            "Enterprise Value": ev,
            "EV/R": evr,
        })

    # yearly
    for dt, r in ydf.iterrows():
        if dt.year >= y0:
            push(dt, r, "yearly", str(dt.year))

    # quarterly (fiscal)
    for dt, r in qdf.iterrows():
        if dt.year >= y0:
            push(dt, r, "quarterly", q_labels.get(dt))

    print("✓")
    time.sleep(0.4)

# ──────────── save Excel (unchanged) ─────────────────────────────────────
if not rows:
    sys.exit("\nNo financial rows fetched — all tickers failed.")

out  = pd.DataFrame(rows).sort_values(["Company Name","period_type","period"])
base = os.path.splitext(path)[0]
fname = f"{base}_financials_{datetime.now():%Y%m%d}.xlsx"
with pd.ExcelWriter(fname, engine="openpyxl") as w:
    out.to_excel(w, sheet_name="Financials", index=False)
    if unresolved:
        pd.DataFrame({"Unresolved": unresolved}).to_excel(w, "Unresolved", index=False)
    if diagnostics:
        pd.DataFrame(diagnostics).to_excel(w, "Diagnostics", index=False)

print(f"\nSaved → {fname}")
if unresolved:
    print("Unresolved companies:", ", ".join(unresolved))
if diagnostics:
    print(f"Diagnostics logged for {len(diagnostics)} statement rows.")
