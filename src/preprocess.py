import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data():
    file_path = "data/unsw_subset.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run Phase 2 first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(file_path)

    # 1. Separate Features (X) and Labels (y)
    # 'id' is a row identifier, 'label' is 0/1, 'attack_cat' is the name of the attack.
    # We drop these so the model doesn't use them to cheat.
    columns_to_drop = ['id', 'label', 'attack_cat']
    drop_cols = [col for col in columns_to_drop if col in df.columns]
    
    y = df['label'] # Save the answer key separately
    X_raw = df.drop(columns=drop_cols)

    # 2. Select only numerical features
    # Isolation Forest needs numbers. We filter out string/text columns.
    X = X_raw.select_dtypes(include=['number'])
    print(f"Selected {X.shape[1]} numerical features out of {df.shape[1]} total columns.")

    # 3. Split into Training and Testing sets
    # 80% training, 20% testing.
    # stratify=y ensures both sets have the same ratio of normal vs. attack traffic.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scale the features (Standardization)
    scaler = StandardScaler()
    
    # IMPORTANT: We only 'fit' (learn the math rules) on the training data!
    # Applying those rules to the test data prevents "data leakage".
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Save the processed data and the scaler
    print("\nSaving preprocessed data and scaler...")
    
    # We MUST save this scaler to a file so our Streamlit dashboard can scale new CSV uploads exactly the same way.
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save the splits to our data folder so Phase 4 can easily load them.
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print("--- Preprocessing Complete ---")
    print(f"Training data shape: {X_train_scaled.shape} (80%)")
    print(f"Testing data shape:  {X_test_scaled.shape} (20%)")
    print("Saved scaler to models/scaler.joblib")
    print("Saved training/testing CSVs to data/ directory")

if __name__ == "__main__":
    preprocess_data()