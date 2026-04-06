# HydroLogic AI: Real-Time Water Potability Monitoring

**HydroLogic AI** is an end-to-end Machine Learning solution designed to classify water safety based on physicochemical sensor metrics. By leveraging advanced data rebalancing techniques and explainable AI, this project provides a transparent, high-recall system for real-time water quality assurance.

## 🚀 Key Features

- **Dual-Model Architecture:** Uses **XGBoost** for efficient edge deployment and **TabNet** (Attention-based Deep Learning) for complex pattern recognition.
- **Intelligent Rebalancing:** Implements **Borderline-SMOTE** to address class imbalance, ensuring high recall for the "Potable" class.
- **Regulatory Transparency:** Integrated **SHAP (SHapley Additive exPlanations)** to provide an audit trail for every safety verdict.
- **Interactive Dashboard:** A Flask-based web interface with real-time **Radar Chart** visualizations of water chemical fingerprints.

## 📊 Dataset Description

The model is trained on the _Water Potability Dataset_, consisting of 3,276 water bodies with the following metrics:

- **pH:** Acid-base balance (WHO Standard: 6.5 - 8.5).
- **Hardness:** Calcium and Magnesium concentration.
- **Solids (TDS):** Total dissolved solids.
- **Chloramines & Sulfate:** Chemical disinfectant and industrial byproduct levels.
- **Conductivity & Turbidity:** Ionic concentration and light-emitting properties.
- **Organic Carbon & Trihalomethanes:** Natural and synthetic organic content.

## 🛠️ Technical Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, XGBoost, PyTorch-TabNet
- **Data Handling:** Pandas, NumPy, Borderline-SMOTE
- **Explainability:** SHAP
- **Frontend:** HTML5, CSS3 (Bootstrap), JavaScript (Chart.js)

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/HydroLogic-AI.git](https://github.com/your-username/HydroLogic-AI.git)
    cd HydroLogic-AI
    ```
2.  **Install dependencies:**
    ```bash
    pip install flask joblib numpy scikit-learn xgboost shap
    ```
3.  **Ensure Model Files are Present:**
    Place your `water_model.pkl` and `scaler.pkl` in the root directory.
4.  **Run the application:**
    ```bash
    python app.py
    ```
5.  **Access the Dashboard:**
    Open `http://127.0.0.1:5000` in your web browser.

## 📈 Model Performance

| Model               | Balanced Accuracy | AUC-ROC  | Recall (Safe Water) |
| :------------------ | :---------------- | :------- | :------------------ |
| Logistic Regression | 52%               | 0.54     | 55%                 |
| XGBoost + B-SMOTE   | **68%**           | **0.71** | **64%**             |
| TabNet              | 66%               | 0.69     | 61%                 |

## 🛡️ Interpretability

We use SHAP summary plots to identify the global impact of features. Our analysis confirms that **Sulfate**, **pH**, and **Solids** are the primary drivers of the model's safety predictions, aligning with WHO drinking water guidelines.

## 🤝 Contributors

- **Rohan Dharmesh Joshi** - [Your GitHub Profile]
- **[Partner Name]** - [Partner GitHub Profile]

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
