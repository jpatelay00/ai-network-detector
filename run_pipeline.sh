#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=================================================="
echo "🛡️ Starting AI Network Anomaly Detector Pipeline"
echo "=================================================="

# 1. Fetch and Explore Data
echo -e "\n>>> STEP 1: Fetching and Exploring Data"
python3 src/explore_data.py

# 2. Preprocess Data
echo -e "\n>>> STEP 2: Preprocessing Data"
python3 src/preprocess.py

# 3. Train Model
echo -e "\n>>> STEP 3: Training Isolation Forest Model"
python3 src/train_model.py

# 4. Evaluate Model
echo -e "\n>>> STEP 4: Evaluating Model Performance"
python3 src/evaluate_model.py

echo -e "\n=================================================="
echo "✅ Pipeline Execution Completed Successfully!"
echo "You can now run 'streamlit run app.py' to launch the dashboard."
echo "=================================================="