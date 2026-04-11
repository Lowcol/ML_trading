import pandas as pd
import os
import glob

def build_master_dataset(processed_dir="data/processed", output_file="data/master_training_data.csv"):
    print(f"Scanning for processed files in {processed_dir}...")
    csv_files = glob.glob(os.path.join(processed_dir, "*_processed.csv"))
    
    if not csv_files:
        print("No processed files found. Please run preprocessing.py first.")
        return

    all_candidates = []

    for file in csv_files:
        # Extract ticker from filename (assuming format like "TSLA_eod_processed.csv")
        filename = os.path.basename(file)
        ticker = filename.split('_')[0].upper()
        
        try:
            df = pd.read_csv(file)
            
            # We only want rows that were flagged as candidates and successfully labeled
            if 'is_candidate' in df.columns and 'label' in df.columns:
                candidates_only = df[(df['is_candidate'] == True) & (df['label'].notna())].copy()
                
                if not candidates_only.empty:
                    candidates_only.insert(0, 'ticker', ticker) # Add ticker as the first column for tracking
                    all_candidates.append(candidates_only)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not all_candidates:
        print("No valid candidate rows found across all files.")
        return

    # Combine everything into one massive DataFrame
    master_df = pd.concat(all_candidates, ignore_index=True)
    
    # Print Statistics
    total_samples = len(master_df)
    successes = int(master_df['label'].sum())
    failures = total_samples - successes
    
    print("\n--- Dataset Build Complete ---")
    print(f"Total labeled candidates: {total_samples}")
    print(f"Label 1 (Successes): {successes} ({(successes/total_samples)*100:.1f}%)")
    print(f"Label 0 (Failures): {failures} ({(failures/total_samples)*100:.1f}%)")
    
    if successes < 500:
        print("\nWarning: You have fewer than 500 success examples.")
        print("Consider processing more tickers in your universe to effectively train a neural network.")

    # Save to disk
    master_df.to_csv(output_file, index=False)
    print(f"\nSaved master dataset to {output_file}")

if __name__ == "__main__":
    build_master_dataset()
