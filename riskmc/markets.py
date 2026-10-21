"""Exchanges, trading hours and a starter ticker list (Yahoo Finance symbols).

Hours are regular sessions in local time; exchange holidays are not modeled.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

MARKETS = [
    {"id": "us", "name": "United States", "exchange": "NYSE / Nasdaq", "tz": "America/New_York",
     "sessions": [("09:30", "16:00")],
     "tickers": [("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"),
                 ("TSLA", "Tesla"), ("JPM", "JPMorgan Chase"), ("SPY", "S&P 500 ETF")]},
    {"id": "ca", "name": "Canada", "exchange": "TSX", "tz": "America/Toronto",
     "sessions": [("09:30", "16:00")],
     "tickers": [("SHOP.TO", "Shopify"), ("RY.TO", "Royal Bank of Canada"), ("ENB.TO", "Enbridge")]},
    {"id": "br", "name": "Brazil", "exchange": "B3", "tz": "America/Sao_Paulo",
     "sessions": [("10:00", "17:00")],
     "tickers": [("PETR4.SA", "Petrobras"), ("VALE3.SA", "Vale"), ("ITUB4.SA", "Itaú Unibanco")]},
    {"id": "uk", "name": "United Kingdom", "exchange": "LSE", "tz": "Europe/London",
     "sessions": [("08:00", "16:30")],
     "tickers": [("SHEL.L", "Shell"), ("AZN.L", "AstraZeneca"), ("HSBA.L", "HSBC"), ("BP.L", "BP")]},
    {"id": "de", "name": "Germany", "exchange": "Xetra", "tz": "Europe/Berlin",
     "sessions": [("09:00", "17:30")],
     "tickers": [("SAP.DE", "SAP"), ("SIE.DE", "Siemens"), ("ALV.DE", "Allianz")]},
    {"id": "fr", "name": "France", "exchange": "Euronext Paris", "tz": "Europe/Paris",
     "sessions": [("09:00", "17:30")],
     "tickers": [("MC.PA", "LVMH"), ("TTE.PA", "TotalEnergies"), ("AIR.PA", "Airbus")]},
    {"id": "ch", "name": "Switzerland", "exchange": "SIX", "tz": "Europe/Zurich",
     "sessions": [("09:00", "17:30")],
     "tickers": [("NESN.SW", "Nestlé"), ("NOVN.SW", "Novartis"), ("ROG.SW", "Roche")]},
    {"id": "in", "name": "India", "exchange": "NSE", "tz": "Asia/Kolkata",
     "sessions": [("09:15", "15:30")],
     "tickers": [("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy"),
                 ("HDFCBANK.NS", "HDFC Bank"), ("INFY.NS", "Infosys")]},
    {"id": "sg", "name": "Singapore", "exchange": "SGX", "tz": "Asia/Singapore",
     "sessions": [("09:00", "12:00"), ("13:00", "17:00")],
     "tickers": [("D05.SI", "DBS Group"), ("O39.SI", "OCBC Bank")]},
    {"id": "hk", "name": "Hong Kong", "exchange": "HKEX", "tz": "Asia/Hong_Kong",
     "sessions": [("09:30", "12:00"), ("13:00", "16:00")],
     "tickers": [("0700.HK", "Tencent"), ("9988.HK", "Alibaba"), ("0005.HK", "HSBC Holdings")]},
    {"id": "kr", "name": "South Korea", "exchange": "KRX", "tz": "Asia/Seoul",
     "sessions": [("09:00", "15:30")],
     "tickers": [("005930.KS", "Samsung Electronics"), ("000660.KS", "SK Hynix")]},
    {"id": "jp", "name": "Japan", "exchange": "Tokyo SE", "tz": "Asia/Tokyo",
     "sessions": [("09:00", "11:30"), ("12:30", "15:30")],
     "tickers": [("7203.T", "Toyota"), ("6758.T", "Sony"), ("9984.T", "SoftBank Group")]},
    {"id": "au", "name": "Australia", "exchange": "ASX", "tz": "Australia/Sydney",
     "sessions": [("10:00", "16:00")],
     "tickers": [("BHP.AX", "BHP Group"), ("CBA.AX", "Commonwealth Bank"), ("CSL.AX", "CSL")]},
    {"id": "fx", "name": "Currencies", "exchange": "FX, 24/5", "tz": "America/New_York",
     "sessions": "fx",
     "tickers": [("EURUSD=X", "Euro / US Dollar"), ("USDJPY=X", "US Dollar / Yen"),
                 ("GBPUSD=X", "Pound / US Dollar")]},
    {"id": "crypto", "name": "Crypto", "exchange": "24/7", "tz": "UTC",
     "sessions": "always",
     "tickers": [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana")]},
]


def _t(s: str) -> time:
    return time.fromisoformat(s)


def status(market: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(market["tz"]))
    sessions = market["sessions"]
    if sessions == "always":
        state = "open"
    elif sessions == "fx":
        # Sunday 17:00 to Friday 17:00 New York time
        wd, t = local.weekday(), local.time()
        closed = wd == 5 or (wd == 4 and t >= _t("17:00")) or (wd == 6 and t < _t("17:00"))
        state = "closed" if closed else "open"
    elif local.weekday() >= 5:
        state = "closed"
    else:
        t = local.time()
        if any(_t(a) <= t < _t(b) for a, b in sessions):
            state = "open"
        elif _t(sessions[0][0]) <= t < _t(sessions[-1][1]):
            state = "break"
        else:
            state = "closed"
    hours = "24/7" if sessions == "always" else "24/5" if sessions == "fx" else \
        ", ".join(f"{a}–{b}" for a, b in sessions)
    return {"state": state, "local_time": local.strftime("%a %H:%M"), "hours": hours}


def catalog() -> list[dict]:
    rows = []
    for m in MARKETS:
        rows.append({
            "id": m["id"], "name": m["name"], "exchange": m["exchange"], **status(m),
            "tickers": [{"symbol": s, "name": n} for s, n in m["tickers"]],
        })
    order = {"open": 0, "break": 1, "closed": 2}
    return sorted(rows, key=lambda r: order[r["state"]])


def market_for(symbol: str) -> dict | None:
    for m in MARKETS:
        if any(s == symbol for s, _ in m["tickers"]):
            return m
    return None
