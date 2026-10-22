import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_isolation_forest():
    train_file = "data/X_train.csv"
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Please ensure Phase 3 completed successfully.")
        return

    print("1. Loading preprocessed training data...")
    X_train = pd.read_csv(train_file)
    print(f"Loaded {X_train.shape[0]} network connections for training.")

    print("\n2. Initializing the Isolation Forest model...")
    # contamination=0.1 tells the model to expect roughly 10% of traffic to be anomalies.
    # n_jobs=-1 tells your MacBook to use all available CPU cores to train faster.
    # random_state=42 ensures the exact same forest is built every time for your demo.
    model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)

    print("\n3. Training the AI... (This will take a few seconds)")
    model.fit(X_train)

    print("\n4. Saving the trained model to disk...")
    model_path = 'models/isolation_forest.joblib'
    joblib.dump(model, model_path)

    print("\n--- Model Training Complete ---")
    print(f"AI Model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_isolation_forest()