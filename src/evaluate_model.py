import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model():
    # Define file paths
    test_file = "data/X_test.csv"
    labels_file = "data/y_test.csv"
    model_file = "models/isolation_forest.joblib"

    # Check if files exist
    if not all(os.path.exists(f) for f in [test_file, labels_file, model_file]):
        print("Error: Missing required files. Ensure Phases 3 and 4 completed successfully.")
        return

    print("1. Loading test data, labels, and the trained model...")
    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv(labels_file)
    model = joblib.load(model_file)

    print(f"Loaded {X_test.shape[0]} test connections.")

    print("\n2. Making predictions on the unseen test data...")
    # The Isolation Forest returns 1 for normal (inliers) and -1 for anomalies (outliers)
    raw_predictions = model.predict(X_test)

    print("3. Formatting predictions to match our dataset labels...")
    # Our dataset labels are: 0 = Normal, 1 = Attack
    # We map: Model's -1 (Outlier) -> 1 (Attack)
    #         Model's 1 (Inlier)  -> 0 (Normal)
    formatted_predictions = [1 if x == -1 else 0 for x in raw_predictions]

    print("\n4. Calculating Evaluation Metrics...")
    print("-" * 50)
    print("Confusion Matrix:")
    # Breakdown:
    # [True Negatives (Normal -> Normal) , False Positives (Normal -> Attack)]
    # [False Negatives (Attack -> Normal) , True Positives (Attack -> Attack)]
    cm = confusion_matrix(y_test, formatted_predictions)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, formatted_predictions, target_names=['Normal (0)', 'Anomaly/Attack (1)']))
    print("-" * 50)
    
    print("\nIMPORTANT CONCEPT:")
    print("Anomalous traffic does NOT necessarily indicate a cyberattack.")
    print("It simply means the traffic behaves differently than the majority.")
    print("Conversely, sophisticated attacks are designed to mimic normal traffic.")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_model()