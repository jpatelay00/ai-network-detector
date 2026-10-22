import pandas as pd
import os
import ssl 

# Tell Python to bypass SSL certificate verification for this script
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_and_explore_data():
    # NEW: Updated URL pointing to an active GitHub mirror of the UNSW-NB15 dataset
    data_url = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
    local_file_path = "data/unsw_subset.csv"
    
    # 1. Download and sample the dataset
    if not os.path.exists(local_file_path):
        print("Downloading dataset from the web... (This might take a minute)")
        try:
            # Read the dataset directly into pandas
            df = pd.read_csv(data_url)
            
            # To respect your 8GB RAM, we sample 10,000 random rows
            df = df.sample(n=10000, random_state=42)
            
            # Save the lightweight subset locally so we don't have to download it again
            df.to_csv(local_file_path, index=False)
            print(f"Successfully saved 10,000 row subset to {local_file_path}")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return
    else:
        print(f"Loading existing dataset from {local_file_path}...")
        df = pd.read_csv(local_file_path)

    # 2. Explore the dataset
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- First 5 Rows (Numerical Features) ---")
    # Select only a few numerical columns to display cleanly in the terminal
    numerical_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
    print(df[numerical_cols].head())

    print("\n--- Traffic Classification Distribution ---")
    # The 'label' column dictates if it is normal (0) or anomalous/attack (1)
    # The 'attack_cat' column tells us the specific type of attack
    print(df['label'].value_counts())
    
    print("\n--- Missing Values Check ---")
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values in dataset: {missing_values}")

if __name__ == "__main__":
    fetch_and_explore_data()