import pandas as pd
import glob
import os

# Assuming your preprocess_breakout_data is imported or in the same file
processed_list = []
raw_files = glob.glob("data/raw/*_eod.csv")
spy_path = "data/raw/spy_eod.csv"

print(f"🔄 Processing {len(raw_files)} tickers into World Model...")

for file in raw_files:
    ticker = os.path.basename(file).split('_')[0]
    if ticker == 'spy': continue # Don't process the index as a candidate
    
    # 1. Process individual ticker
    df = preprocess_breakout_data(file, spy_file_path=spy_path)
    
    # 2. Add ticker identity (useful for debugging, though model should ignore it)
    df['symbol'] = ticker 
    
    # 3. Save modular processed file (Individual)
    df.to_parquet(f"data/processed/{ticker}_processed.parquet")
    
    # 4. Extract ONLY the candidates for the Giant File (Training)
    candidates = df[df['is_candidate'] == True].copy()
    processed_list.append(candidates)

# 5. Create the "Giant" Training File
master_df = pd.concat(processed_list, ignore_index=True)

# 6. Final Shuffling (Crucial for World Models)
# This breaks the chronological/ticker bias
master_df = master_df.sample(frac=1).reset_index(drop=True)

master_df.to_parquet("data/master_candidates_training.parquet")
print(f"✅ Created World Model with {len(master_df)} total candidates.")