# 🚀 Boost — Smart Scoring for the Informal Sector

**Boost** is a data-driven microcredit risk assessment prototype designed for informal and underbanked economies.  
Instead of relying only on traditional credit history, Boost leverages machine learning proxies to estimate financial readiness and default risk, empowering both lenders (banks) and borrowers (customers).

> **Tagline:** *Predicting potential, not just history.*

---

## 🧠 Project Overview

Boost is implemented as an interactive Streamlit application that combines:
* A trained machine learning credit risk model.
* Portfolio-level decision support for lenders.
* An intuitive, user-facing eligibility experience for customers.

The application is structured around two complementary views:
1.  **🏦 Bank View** — Portfolio pre-screening & decision support.
2.  **👤 Customer View** — Individual eligibility & simulation.

---

## 🖥️ Streamlit Application

The Streamlit app (`streamlitapp.py`) is the main entry point. It loads the trained model, processes data samples, and renders the interface.

**Key responsibilities:**
* Load the trained model and selected features.
* Compute default risk and eligibility scores.
* Render interactive dashboards for banks and customers.
* Share parameters (e.g., eligibility threshold) across views.

---

## 🏦 Bank View — Portfolio Pre-Screening

Designed for lenders and risk teams to explore risk–reward trade-offs.

**Functionality:**
* **Adjustable Threshold:** Set the Eligibility Score (1–100) dynamically.
* **KPIs:** Real-time count of approved/rejected applicants and average default risk.
* **Visualizations:** * Interactive table of applicants.
    * Eligibility score distribution histogram with approval threshold lines.

---

## 👤 Customer Financial Health View

Focuses on transparency and empowerment for borrowers, split into two sub-tabs:

### 👤 Account
* **Eligibility Score:** View individual score and risk band (High / Medium / Low).
* **Dynamic Feedback:** Status updates based on the bank's current threshold.
* **Action-Oriented Messaging:** Eligible users can "Apply", while others receive personalized improvement tips.

### 🧪 Simulation (Sandbox)
* **What-if Analysis:** Users can modify inputs (income, household size, stability) to see how their score updates live.
* **Education:** Helps users understand what drives their score and how to improve it.

---

## 🤖 Machine Learning Model

### Dataset
The model is trained on the **Home Credit Default Risk** dataset (application_train.csv). It is ideal for this project as it assesses creditworthiness beyond traditional bureau data.
[Link Here:](https://www.kaggle.com/competitions/home-credit-default-risk)

### Training
Documented in `📓 Model_Training.ipynb`, covering:
* Data cleaning and feature selection.
* Categorical encoding and scaling.
* **Logistic Regression** training and ROC-AUC evaluation.

### Model Artifacts
The final model is saved in:
`Models/microcredit_model.joblib`

---

## 📂 Project Structure

```text
prototype_1/
│
├── streamlitapp.py          # Main Streamlit application
├── Model_Training.ipynb     # Model training & evaluation
├── README.md                # Documentation
│
├── DATA/
│   └── application_train.csv
│
├── Models/
│   └── microcredit_model.joblib
│
└── .streamlit/
    └── config.toml          # UI theme configuration
```

## 📦 Required Libraries

Install the dependencies using pip:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly
```

## ▶️ How to Run the App

```bash
From the project root directory, run:

```bash
streamlit run streamlitapp.py
```

## 🎯 Purpose & Disclaimer
This project is a prototype and educational showcase. It demonstrates how AI can support inclusive finance but is not a production-ready credit decision system.