# 🎓 Student Dropout Risk Predictor
### AI-Powered Early Warning System for Educational Institutions

A production-ready machine learning solution that identifies at-risk 
students and provides actionable, explainable insights for timely 
academic intervention.

---

## 🚨 The Problem

Educational institutions often identify dropout risks too late — after 
a student has already disengaged. This system analyzes hidden patterns 
in student data (attendance trends, academic scores, geographic distance) 
allowing administrators to intervene **weeks before a student leaves.**

---

## ✨ Key Features

- **Risk Scoring** — Classifies students as High Risk or Low Risk using 
  a class-balanced Random Forest model
- **SHAP Explainability** — Explains *why* each student is flagged 
  (e.g. high distance + low attendance = intervention needed)
- **Optimized Metrics** — Evaluated using PR-AUC for reliable 
  performance on imbalanced datasets
- **Visual Analytics** — Confusion matrices, ROC curves, and feature 
  importance rankings for administrative reporting
- **Interactive Dashboard** — Live Streamlit app with sidebar controls 
  and real-time SHAP explanations

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Model | Scikit-learn Random Forest |
| Explainability | SHAP (Shapley Additive Explanations) |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Data | Pandas, NumPy |
| Deployment | Pickle + ngrok |

---

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/imadali/student-dropout-predictor

2. Install dependencies
   pip install -r requirements.txt

3. Train the model
   jupyter notebook Day29_Capstone_StudentDashboard.ipynb

4. Launch the dashboard
   streamlit run capstone_app.py

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 1.000 |
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| Recall (Dropout) | 1.000 |

---

## 👨‍💻 Author

**Imad Ali**
Data Scientist | ML Engineer | Python Developer
📍 KPK, Pakistan
⭐ [Fiverr Profile](https://fiverr.com) — 5.0 rated, 43 reviews
💼 Available for freelance projects

---

*Built as part of DS/ML University — a structured self-directed 
machine learning program.*