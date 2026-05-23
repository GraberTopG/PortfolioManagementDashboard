"""
config.py - Dashboard constants.
==================================
All project-wide constants: annualisation factor, risk-free rate,
ticker universe, colour palette, and chart styling variables.
No project imports; safe to import from any other module.
"""

# ── Annualisation & risk-free rate ────────────────────────────────────────────
AF = 252
RF = 0.0525

# ── Default / benchmark tickers ──────────────────────────────────────────────
DEFAULT_TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS"]
BENCHMARK_TICKERS = ["SPY", "AGG"]

# ── Stock universe organised by GICS sector ──────────────────────────────────
UNIVERSE = {
    "Technology": [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","AMD","ORCL",
        "CRM","ADBE","INTC","QCOM","AVGO","TXN","NOW","PANW","INTU","CSCO",
        "IBM","ACN","AMAT","MU","KLAC","LRCX","SNPS","CDNS","FTNT","HPQ",
    ],
    "Financials": [
        "JPM","GS","BAC","MS","BLK","V","MA","C","WFC","AXP",
        "SCHW","SPGI","ICE","CME","MCO","PGR","AON","MET","TFC","USB",
        "PNC","COF","DFS","FITB","KEY",
    ],
    "Healthcare": [
        "JNJ","UNH","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD","MDT",
        "TMO","ABT","ISRG","SYK","BSX","ELV","CVS","HUM","CI","DHR",
        "VRTX","REGN","ZBH","BDX","IQV",
    ],
    "Consumer Discretionary": [
        "MCD","NKE","TSCO","HD","LOW","SBUX","TGT","CMG","YUM","BKNG",
        "MAR","HLT","F","GM","ORLY","AZO","ROST","TJX","VFC","PVH",
    ],
    "Consumer Staples": [
        "WMT","KO","PG","PEP","COST","CL","KMB","GIS","MO","PM",
        "MDLZ","KHC","SYY","CHD","CAG",
    ],
    "Industrials": [
        "BA","CAT","HON","GE","LMT","RTX","DE","UNP","UPS","FDX",
        "MMM","EMR","ETN","GD","NOC","PH","ROK","TDG","CTAS","RSG",
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","EOG","MPC","OXY","HAL","PSX","VLO",
        "HES","DVN","BKR","FANG","MRO",
    ],
    "Communication": [
        "T","VZ","CMCSA","DIS","CHTR","WBD","PARA","FOXA","OMC","IPG",
    ],
    "Real Estate": [
        "AMT","PLD","CCI","EQIX","SPG","O","WELL","AVB","EQR","PSA",
    ],
    "Materials": [
        "LIN","APD","SHW","NEM","FCX","NUE","PPG","ALB","CF","MOS",
    ],
    "Utilities": [
        "NEE","DUK","SO","AEP","D","EXC","PCG","XEL","ES","AWK",
    ],
}
ALL_TICKERS   = [t for group in UNIVERSE.values() for t in group]
TICKER_SECTOR = {t: sec for sec, tickers in UNIVERSE.items() for t in tickers}

# Approximate S&P 500 GICS sector weights (as of 2025)
SPY_SECTOR_WEIGHTS = {
    "Technology": 0.310, "Financials": 0.134, "Healthcare": 0.118,
    "Consumer Discretionary": 0.104, "Communication": 0.090,
    "Industrials": 0.088, "Consumer Staples": 0.058, "Energy": 0.038,
    "Utilities": 0.024, "Real Estate": 0.022, "Materials": 0.022,
}

# ── Chart styling ─────────────────────────────────────────────────────────────
CHART_TEMPLATE = "plotly_dark"
# Bloomberg Terminal colour palette
ACCENT = "#FF8C00"   # Bloomberg orange  (primary highlight)
BLUE   = "#00A8E8"   # Bloomberg blue    (secondary)
RED    = "#E53935"   # Bloomberg red     (losses / risk)
GREEN  = "#00C853"   # Bloomberg green   (gains / positive)
GOLD   = "#FFD600"   # Bloomberg yellow  (caution / neutral)
MUTED  = "#78909C"   # blue-grey         (secondary text)

# ── Typography & background colours ──────────────────────────────────────────
_FONT  = "IBM Plex Serif, Georgia, serif"
_MONO  = "IBM Plex Mono, Courier New, monospace"
_SERIF = "IBM Plex Serif, Georgia, serif"
_BG    = "#0A0C10"   # paper / outer background (Bloomberg near-black)
_PLOT  = "#0E1117"   # inner plot area
_GRID  = "#181D24"   # gridlines (very subtle)
_AXIS  = "#263238"   # axis lines
_TICK  = "#546E7A"   # tick labels

# ── Optimisation cap ──────────────────────────────────────────────────────────
# Maximum weight any single asset may receive in optimised portfolios.
# Prevents the solver from crowding into one stock that happened to dominate
# the historical sample (e.g. NVDA post-2020). 40% is a standard long-only cap.
MAX_SINGLE_W = 0.40

# ── Company display names ─────────────────────────────────────────────────────
COMPANY_NAMES = {
    # Technology
    "AAPL":"Apple",                 "MSFT":"Microsoft",           "GOOGL":"Alphabet",
    "AMZN":"Amazon",                "META":"Meta Platforms",       "NVDA":"NVIDIA",
    "TSLA":"Tesla",                 "NFLX":"Netflix",              "AMD":"AMD",
    "ORCL":"Oracle",                "CRM":"Salesforce",            "ADBE":"Adobe",
    "INTC":"Intel",                 "QCOM":"Qualcomm",             "AVGO":"Broadcom",
    "TXN":"Texas Instruments",      "NOW":"ServiceNow",            "PANW":"Palo Alto Networks",
    "INTU":"Intuit",                "CSCO":"Cisco",                "IBM":"IBM",
    "ACN":"Accenture",              "AMAT":"Applied Materials",    "MU":"Micron Technology",
    "KLAC":"KLA Corp",              "LRCX":"Lam Research",         "SNPS":"Synopsys",
    "CDNS":"Cadence Design",        "FTNT":"Fortinet",             "HPQ":"HP Inc.",
    # Financials
    "JPM":"JPMorgan Chase",         "GS":"Goldman Sachs",          "BAC":"Bank of America",
    "MS":"Morgan Stanley",          "BLK":"BlackRock",             "V":"Visa",
    "MA":"Mastercard",              "C":"Citigroup",               "WFC":"Wells Fargo",
    "AXP":"American Express",       "SCHW":"Charles Schwab",       "SPGI":"S&P Global",
    "ICE":"Intercontinental Exch.", "CME":"CME Group",             "MCO":"Moody's",
    "PGR":"Progressive",            "AON":"Aon",                   "MET":"MetLife",
    "TFC":"Truist Financial",       "USB":"U.S. Bancorp",          "PNC":"PNC Financial",
    "COF":"Capital One",            "DFS":"Discover Financial",    "FITB":"Fifth Third Bancorp",
    "KEY":"KeyCorp",
    # Healthcare
    "JNJ":"Johnson & Johnson",      "UNH":"UnitedHealth",          "PFE":"Pfizer",
    "ABBV":"AbbVie",                "MRK":"Merck",                 "LLY":"Eli Lilly",
    "BMY":"Bristol-Myers Squibb",   "AMGN":"Amgen",                "GILD":"Gilead Sciences",
    "MDT":"Medtronic",              "TMO":"Thermo Fisher",         "ABT":"Abbott Labs",
    "ISRG":"Intuitive Surgical",    "SYK":"Stryker",               "BSX":"Boston Scientific",
    "ELV":"Elevance Health",        "CVS":"CVS Health",            "HUM":"Humana",
    "CI":"Cigna",                   "DHR":"Danaher",               "VRTX":"Vertex Pharma.",
    "REGN":"Regeneron",             "ZBH":"Zimmer Biomet",         "BDX":"Becton Dickinson",
    "IQV":"IQVIA",
    # Consumer Discretionary
    "MCD":"McDonald's",             "NKE":"Nike",                  "TSCO":"Tractor Supply",
    "HD":"Home Depot",              "LOW":"Lowe's",                "SBUX":"Starbucks",
    "TGT":"Target",                 "CMG":"Chipotle",              "YUM":"Yum! Brands",
    "BKNG":"Booking Holdings",      "MAR":"Marriott",              "HLT":"Hilton",
    "F":"Ford",                     "GM":"General Motors",         "ORLY":"O'Reilly Auto Parts",
    "AZO":"AutoZone",               "ROST":"Ross Stores",          "TJX":"TJX Companies",
    "VFC":"VF Corp",                "PVH":"PVH Corp",
    # Consumer Staples
    "WMT":"Walmart",                "KO":"Coca-Cola",              "PG":"Procter & Gamble",
    "PEP":"PepsiCo",                "COST":"Costco",               "CL":"Colgate-Palmolive",
    "KMB":"Kimberly-Clark",         "GIS":"General Mills",         "MO":"Altria",
    "PM":"Philip Morris",           "MDLZ":"Mondelez",             "KHC":"Kraft Heinz",
    "SYY":"Sysco",                  "CHD":"Church & Dwight",       "CAG":"Conagra Brands",
    # Industrials
    "BA":"Boeing",                  "CAT":"Caterpillar",           "HON":"Honeywell",
    "GE":"GE Aerospace",            "LMT":"Lockheed Martin",       "RTX":"RTX Corp",
    "DE":"Deere & Co",              "UNP":"Union Pacific",         "UPS":"UPS",
    "FDX":"FedEx",                  "MMM":"3M",                    "EMR":"Emerson Electric",
    "ETN":"Eaton",                  "GD":"General Dynamics",       "NOC":"Northrop Grumman",
    "PH":"Parker Hannifin",         "ROK":"Rockwell Automation",   "TDG":"TransDigm",
    "CTAS":"Cintas",                "RSG":"Republic Services",
    # Energy
    "XOM":"ExxonMobil",             "CVX":"Chevron",               "COP":"ConocoPhillips",
    "SLB":"SLB",                    "EOG":"EOG Resources",         "MPC":"Marathon Petroleum",
    "OXY":"Occidental Petroleum",   "HAL":"Halliburton",           "PSX":"Phillips 66",
    "VLO":"Valero Energy",          "HES":"Hess",                  "DVN":"Devon Energy",
    "BKR":"Baker Hughes",           "FANG":"Diamondback Energy",   "MRO":"Marathon Oil",
    # Communication
    "T":"AT&T",                     "VZ":"Verizon",                "CMCSA":"Comcast",
    "DIS":"Walt Disney",            "CHTR":"Charter Comm.",        "WBD":"Warner Bros. Discovery",
    "PARA":"Paramount Global",      "FOXA":"Fox Corp",             "OMC":"Omnicom",
    "IPG":"Interpublic Group",
    # Real Estate
    "AMT":"American Tower",         "PLD":"Prologis",              "CCI":"Crown Castle",
    "EQIX":"Equinix",               "SPG":"Simon Property Group",  "O":"Realty Income",
    "WELL":"Welltower",             "AVB":"AvalonBay Communities", "EQR":"Equity Residential",
    "PSA":"Public Storage",
    # Materials
    "LIN":"Linde",                  "APD":"Air Products",          "SHW":"Sherwin-Williams",
    "NEM":"Newmont",                "FCX":"Freeport-McMoRan",      "NUE":"Nucor",
    "PPG":"PPG Industries",         "ALB":"Albemarle",             "CF":"CF Industries",
    "MOS":"Mosaic",
    # Utilities
    "NEE":"NextEra Energy",         "DUK":"Duke Energy",           "SO":"Southern Company",
    "AEP":"American Electric Power","D":"Dominion Energy",         "EXC":"Exelon",
    "PCG":"PG&E",                   "XEL":"Xcel Energy",           "ES":"Eversource Energy",
    "AWK":"American Water Works",
}
