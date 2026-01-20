# 🔐 UPI Fraud Detection System

## 📌 Project Overview

This project implements an **end-to-end UPI Fraud Detection System** using supervised machine learning to identify fraudulent transactions. It addresses real-world challenges such as **class imbalance**, **threshold optimization**, and **deployment-ready inference**.

The solution includes data generation, preprocessing, model training, evaluation, threshold tuning, and deployment through an interactive **Streamlit web application**.

---

## 🚀 Features

* Synthetic UPI transaction data generation
* Feature engineering from timestamps
* Class imbalance handling
* Multiple ML models comparison
* Hyperparameter & threshold tuning
* Confusion matrix–based evaluation
* Interactive Streamlit UI with fraud alerts
* Production-ready model artifact (`.pkl`)

---

## 🧠 Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Balanced Random Forest
* **EasyEnsembleClassifier (Final Model)**

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1-score
* Confusion Matrix

(Accuracy was not prioritized due to class imbalance.)

---

## 🏆 Final Model

* **Model:** EasyEnsembleClassifier
* **Threshold:** 0.46
* Optimized for **fraud recall and F1-score**

---

## 🖥️ Deployment

* Built using **Streamlit**
* Two-page interactive UI (Input → Result)
* Color-coded fraud/non-fraud screens
* Real-time prediction with risk visualization

Run the app:

```bash
python3 -m streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py
├── easyensemble_fraud_model.pkl
├── upi_fraud_data.csv
├── UPI_Fraud_Detection.ipynb
├── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* Streamlit


---

## 🔮 Future Improvements

* Real UPI transaction data
* SHAP-based explainability
* Cost-sensitive learning
* Cloud deployment (AWS/GCP)
