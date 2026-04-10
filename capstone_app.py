
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Risk Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ── Load Model and Encoder ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("capstone_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("capstone_encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    return model, enc

model, enc = load_model()

feature_names = ["Attendance Rate", "Avg Score", "Distance (km)",
                 "Family Income", "Parent Education"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎓 Student Dropout Risk Dashboard")
st.markdown("**Powered by Random Forest + SHAP Explainability**")
st.markdown("---")

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("📋 Student Profile")
attendance = st.sidebar.slider("Attendance Rate (%)", 20, 100, 70)
score = st.sidebar.slider("Average Score", 20, 100, 65)
distance = st.sidebar.slider("Distance from School (km)", 1, 40, 10)
income = st.sidebar.selectbox("Family Income", ["low", "medium", "high"])
education = st.sidebar.selectbox("Parent Education",
                                  ["none", "primary", "secondary", "higher"])

# ── Prediction ─────────────────────────────────────────────────────────────────
input_df = pd.DataFrame({
    "attendance_rate": [attendance],
    "avg_score": [score],
    "distance_km": [distance],
    "family_income": [income],
    "parent_education": [education]
})

input_df[["family_income_encoded", "parent_education_encoded"]] = enc.transform(
    input_df[["family_income", "parent_education"]]
)

X_input = input_df[["attendance_rate", "avg_score", "distance_km",
                     "family_income_encoded", "parent_education_encoded"]]

prediction = model.predict(X_input)[0]
probability = model.predict_proba(X_input)[0]
dropout_prob = probability[1]

# ── Results Row ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dropout Risk", f"{dropout_prob:.1%}")

with col2:
    if prediction == 1:
        st.error("⚠️ HIGH RISK — Intervention Needed")
    else:
        st.success("✅ LOW RISK — Student On Track")

with col3:
    st.metric("Attendance", f"{attendance}%")

st.markdown("---")

# ── Two Column Layout ──────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.barh([feature_names[i] for i in indices[::-1]],
             importances[indices[::-1]],
             color="#3498db", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Importance Score")
    ax1.set_title("Global Feature Importance")
    ax1.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with right:
    st.subheader("🔍 SHAP Explanation — This Student")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    shap_vals = shap_values[0, :, 1]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_vals]
    ax2.barh(feature_names, shap_vals, color=colors, edgecolor="black", alpha=0.8)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP Value (red = increases risk)")
    ax2.set_title("Why This Prediction?")
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built by Imad Ali | DS/ML Freelancer | KPK, Pakistan")
