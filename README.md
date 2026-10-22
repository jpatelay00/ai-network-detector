# 🛡️ AI Network Anomaly Detector

**Author:** Jay R. Patel  
**Institution:** Georgia State University | Artificial Intelligence Final Project  

## 📌 Project Objective
A Python-based machine-learning application that analyzes network traffic datasets to identify potentially anomalous connections. This project demonstrates how unsupervised artificial intelligence (Isolation Forests) can assist network engineers and IT professionals in identifying unusual network behavior without relying on predefined attack signatures.

## 🚀 Features
- **Automated Data Pipeline:** Fetches, samples, and preprocesses the UNSW-NB15 network intrusion dataset.
- **Unsupervised Learning:** Utilizes an `IsolationForest` model to detect outliers in high-dimensional network traffic.
- **Interactive Dashboard:** A Streamlit web application providing live metric tracking, traffic classification charts, and anomalous data extraction.
- **Reproducible Build:** Includes a master Bash script (`run_pipeline.sh`) to execute the entire data pipeline from scratch.

## 🛠️ Technology Stack
- **Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Isolation Forest)
- **Web Dashboard:** Streamlit
- **Model Serialization:** Joblib

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ai-network-detector.git](https://github.com/your-username/ai-network-detector.git)
   cd ai-network-detector