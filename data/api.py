import os
import time
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

API_TOKEN = os.getenv("TIINGO_API_TOKEN", "")
START_DATE = "2020-01-01"
DATA_DIR = "data/raw"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class TiingoTraderData:
    def __init__(self, token):
        self.token = token
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {token}'
        }

    def get_historical_eod(self, ticker, start_date, retries=3):
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {'startDate': start_date, 'format': 'json'}
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=self.headers)
                if response.status_code == 200:
                    df = pd.DataFrame(response.json())
                    cols = ['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']
                    return df[cols]
                elif response.status_code == 429:
                    print(f"  [!] Rate limit (429) hit for {ticker}. Sleeping for 60 seconds... (Attempt {attempt+1}/{retries})")
                    time.sleep(60)
                else:
                    print(f"Error fetching {ticker}: {response.status_code}")
                    break
            except Exception as e:
                print(f"Exception for {ticker}: {e}")
                time.sleep(5)
        return pd.DataFrame()

    def get_intraday_trigger(self, ticker, start_date, freq='1min'):
        url = f"https://api.tiingo.com/iex/{ticker}/prices"
        params = {
            'startDate': start_date,
            'resampleFreq': freq,
            'columns': 'open,high,low,close,volume'
        }
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
        except Exception as e:
            print(f"Exception for {ticker} intraday: {e}")
        return pd.DataFrame()

# --- Ticker Selection ---
TICKER_LIST = {
    # 1. THE AI & SEMI REVOLUTION (2023-2024 Alpha)
    # "AI_SEMI": [
    #     "NVDA", "AMD", "SMCI", "AVGO", "TSM", "ARM", "VRT", "ANET", "MU", "MRVL", 
    #     "ASML", "LRCX", "KLAC", "SNPS", "CDNS", "VRT", "DELL", "HPE"
    # ],
    # # 2. SOFTWARE, SAAS & CYBERSECURITY (High Momentum)
    "SAAS_CYBER": [
        # "PLTR", "SNOW", "SHOP", "MDB", "TEAM", "DDOG", "NET", "CRWD", "OKTA", 
        # "ZS", "PANW", "HUBS", "PCOR", "DOCN", "CFLT", "GTLB", "PATH", "AI", 
        "MSFT"
    ],
    # 3. THE "VOLATILITY FACTORY" (Biotech & MedTech)
    # These provide the most "Label 1" and "Label 0" examples for the model
    "BIOTECH_MED": [
        "LLY", "NVO", "VRTX", "REGN", "ISRG", "MRNA", "BNTX", "CRSP", "BEAM", 
        "NTLA", "EDIT", "SAVA", "KOD", "BMEA", "IMTX", "AXSM", "VKTX", "ALT"
    ],
    # 4. FINTECH, CRYPTO & GAMING (Sentiment Driven)
    "FINTECH_CRYPTO": [
        "COIN", "SQ", "PYPL", "MARA", "RIOT", "CLSK", "AFRM", "SOFI", "NU", 
        "HOOD", "UPST", "MQ", "DKNG", "PENN", "RBLX"
    ],
    # 5. E-COMMERCE & GLOBAL GROWTH (High Beta)
    "GLOBAL_GROWTH": [
        "MELI", "SE", "AMZN", "BABA", "PDD", "DASH", "ABNB", "BKNG", "CHWY", 
        "CPNG", "ETSY", "JD", "TME", "LI", "NIO", "XPEV"
    ],
    # 6. ENERGY, URANIUM & STRATEGIC METALS (The 2022 Cycle)
    "ENERGY_METALS": [
        "CCJ", "URA", "UUUU", "NXE", "DNN", "LAC", "ALB", "SQM", "MP", "ENPH", 
        "SEDG", "RUN", "FSLR", "XME", "FCX", "CLF"
    ],
    # 7. INDUSTRIAL MOMENTUM & INFRASTRUCTURE
    "INDUSTRIALS": [
        "CAT", "DE", "URI", "PWR", "EME", "FIX", "ETN", "PH", "GWW", "AXON", "TDG"
    ],
    # 8. THE "WILD CARDS" (EVs, Space, and High Beta Runners)
    "HIGH_BETA": [
        "TSLA", "RIVN", "LCID", "SPCE", "U", "TWLO", "SNAP", "RCL", "CCL", "NCLH", 
        "DUOL", "SMCT", "IONQ", "PLUG", "HOOD"
    ],
    # 9. MARKET REGIME (Keep these for context)
    "REGIME": ["SPY", "QQQ", "IWM", "ARKK", "BITO", "GLD"]
}

if __name__ == "__main__":
    if not API_TOKEN:
        raise ValueError("Missing TIINGO_API_TOKEN environment variable.")

    pipeline = TiingoTraderData(API_TOKEN)
    
    # Flatten the ticker list for the loop
    all_tickers = [t for sublist in TICKER_LIST.values() for t in sublist]
    
    print(f"🚀 Starting database build for {len(all_tickers)} tickers...")

    for ticker in all_tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # 1. Fetch EOD Data
        eod_data = pipeline.get_historical_eod(ticker, START_DATE)
        if not eod_data.empty:
            eod_path = os.path.join(DATA_DIR, f"{ticker.lower()}_eod.csv")
            eod_data.to_csv(eod_path, index=False)
            print(f"✓ Saved EOD: {eod_path}")
        

        # 2. Rate Limiting: Tiingo limits free users to e.g. 50 requests/min and 500/hour.
        # We increase the sleep to keep it safely below the ~50 request per minute limit.
        time.sleep(1.5) 

    print("\n✅ Database build complete. Ready for preprocessing.")