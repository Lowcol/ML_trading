import pandas as pd
import numpy as np
import os
import glob


def add_breakout_labels(df, lookahead_days=10, target_gain=0.20, stop_loss=0.05):
    """Label candidates as success (1) or failure (0) based on forward price path."""
    df = df.copy()
    df['is_candidate'] = (
        (df['mom_90d'] > 0.30) & 
        (df['vol_contraction'] < 0.75) & 
        (df['vol_dryup'] < 0.80) &
        (df['market_regime'] == 1) # Only trade when SPY > EMA200
    )
    df['label'] = np.nan

    for idx in df.index[df['is_candidate']]:
        entry_price = df.at[idx, 'adjClose']
        target_price = entry_price * (1 + target_gain)
        stop_price = entry_price * (1 - stop_loss)

        future = df.loc[idx + 1: idx + lookahead_days, ['adjHigh', 'adjLow']]

        if future.empty:
            continue

        label = np.nan
        for _, row in future.iterrows():
            hit_target = row['adjHigh'] >= target_price
            hit_stop = row['adjLow'] <= stop_price

            # Conservative tie-break: if both are hit in the same bar, mark as failure.
            if hit_stop:
                label = 0
                break
            if hit_target:
                label = 1
                break

        if not np.isnan(label):
            df.at[idx, 'label'] = label

    return df


def normalize_features(df, feature_cols, window=252):
    """Rolling Z-score normalize selected feature columns only to prevent data leakage."""
    df = df.copy()
    for col in feature_cols:
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()
        norm_col = f"{col}_norm"
        
        # Avoid division by zero and fill NaNs (due to warm-up or zero std)
        df[norm_col] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        df[norm_col] = df[norm_col].fillna(0.0)
    return df


def preprocess_breakout_data(file_path, spy_file_path=None):
    df = pd.read_csv(file_path)
    
    # 1. Ensure time-sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 1.5. Market Regime (SPY > 200 EMA)
    if spy_file_path:
        spy_df = pd.read_csv(spy_file_path)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.sort_values('date')
        spy_ema200 = spy_df['adjClose'].ewm(span=200, adjust=False).mean()
        spy_df['market_regime'] = (spy_df['adjClose'] > spy_ema200).astype(int)
        
        # We rename SPY's adjClose to avoid overlapping with the stock's adjClose
        spy_df = spy_df.rename(columns={'adjClose': 'spy_adjClose'})
        
        df = pd.merge(df, spy_df[['date', 'market_regime', 'spy_adjClose']], on='date', how='left')
        # Forward-fill missing dates (e.g. holidays mismatch), fallback to 1
        df['market_regime'] = df['market_regime'].ffill().fillna(1).astype(int)
        
        # Calculate how much the stock is outperforming the index over 3 months
        # We forward-fill spy_adjClose in case of trading holiday mismatches before calculating pct_change
        df['spy_mom_90d'] = df['spy_adjClose'].ffill().pct_change(periods=90)
    else:
        df['market_regime'] = 1
        df['spy_mom_90d'] = 0.0

    # 2. Momentum: The "30-100% in 1-3 months" filter
    df['mom_90d'] = df['adjClose'].pct_change(periods=90)
    df['mom_30d'] = df['adjClose'].pct_change(periods=30)
    
    # Simple RS: Stock Momentum minus Index Momentum
    df['relative_strength_90d'] = df['mom_90d'] - df['spy_mom_90d'].fillna(0.0)

    # 3. Moving Averages (The "Surfing" Logic)
    df['ema10'] = df['adjClose'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['adjClose'].ewm(span=20, adjust=False).mean()
    
    # 4. Volatility Contraction (Tightness)
    # ADR_pct: Average Daily Range as a percentage
    df['daily_range'] = (df['adjHigh'] - df['adjLow']) / df['adjLow']
    df['adr_20d'] = df['daily_range'].rolling(window=20).mean()
    
    # "Tightness" is when current volatility is lower than historical
    df['vol_contraction'] = df['daily_range'].rolling(window=5).mean() / df['adr_20d']

    # 5. Relative Volume (The "Confirmation")
    df['avg_vol_20d'] = df['adjVolume'].rolling(window=20).mean()
    df['rel_vol'] = df['adjVolume'] / df['avg_vol_20d']
    
    # Volume Dry-up: Is the current 5-day volume lower than the 20-day average?
    df['vol_dryup'] = df['adjVolume'].rolling(window=5).mean() / df['avg_vol_20d']

    # 6. Scale-invariant distance features (not raw price normalization)
    df['pct_from_ema10'] = (df['adjClose'] / df['ema10']) - 1
    df['pct_from_ema20'] = (df['adjClose'] / df['ema20']) - 1

    # Remove warm-up rows from rolling calculations before labeling/normalizing.
    feature_cols = [
        'mom_90d',
        'mom_30d',
        'vol_contraction',
        'vol_dryup',
        'rel_vol',
        'pct_from_ema10',
        'pct_from_ema20',
        'relative_strength_90d',  # Adding RS to the feature list
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # 7. Supervised labels for classifier pretraining
    df = add_breakout_labels(df)

    # 8. Normalize feature columns for model input (raw prices are kept unchanged).
    df = normalize_features(df, feature_cols)

    # Ensure 'market_regime' is in the final list of columns for your NumPy tensor.
    # Expose this attribute so downstream scripts know exactly what columns to pull.
    df.attrs['final_model_inputs'] = [f"{c}_norm" for c in feature_cols] + ['market_regime']

    return df

if __name__ == "__main__":
    # Directory setup
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    spy_path = os.path.join(raw_dir, "SPY_daily.csv")  # Replace with your actual SPY file name in raw/
    if not os.path.exists(spy_path):
        print(f"Warning: SPY data not found at {spy_path}. Regime features will default to 1.")
        spy_path = None

    # Process all CSV files in the raw folder
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    for input_path in csv_files:
        # Skip the SPY file itself if we don't want to process it as a target stock
        if "spy" in os.path.basename(input_path).lower():
            continue
            
        filename = os.path.basename(input_path)
        output_path = os.path.join(processed_dir, filename.replace(".csv", "_processed.csv"))
        
        try:
            print(f"Processing {filename}...")
            processed_df = preprocess_breakout_data(input_path, spy_file_path=spy_path)
            processed_df.to_csv(output_path, index=False)
            
            print(f"  Processed rows: {len(processed_df)}")
            print(f"  Candidates: {int(processed_df['is_candidate'].sum())}")
            print(f"  Labeled candidates: {int(processed_df['label'].notna().sum())}")
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            
    print("Done processing all files.")