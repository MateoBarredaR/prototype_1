import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MicroMate: Smart scoring for the informal sector.", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "microcredit_model.joblib")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selected_features"]

clf, selected_features = load_model()

def risk_to_score(risk_prob: float) -> float:
    return round(100 * (1 - risk_prob), 1)

st.title("MicroMate")
st.subheader("Predicting potential, not just history")
st.markdown(
    """
    MicroMate is a microcredit risk assessment tool designed for the informal sector. 
    It combines machine learning with financial health insights to empower both lenders and borrowers.
    """
)
st.caption("Two perspectives: Bank decisioning and Customer financial health.")

tab_bank, tab_customer = st.tabs(["Bank View", "Customer Financial Health View"])

#Sampling Data (for demo purposes)
@st.cache_data
def load_data_sample(n=5000, seed=42):
    df = pd.read_csv(os.path.join("DATA", "application_train.csv"))
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df

# Bank View
with tab_bank:
    st.subheader("Portfolio pre-screening (Bank View)")

    col1, col2, col3 = st.columns(3)
    with col1:
        score_threshold = st.slider("Eligibility score threshold", 1, 100, 80, 1)
    with col2:
        sample_size = st.selectbox("Sample size", [1000, 3000, 5000, 10000], index=2)
    with col3:
        max_rows = st.number_input("Rows to display (table)", value=25, min_value=5, max_value=10000)

    # Load sample
    df_sample = load_data_sample(n=sample_size)

    # Predict risk + score for the whole sample
    X_bank = df_sample[selected_features].copy()
    risk_probs = clf.predict_proba(X_bank)[:, 1]
    df_sample["risk_prob"] = risk_probs
    df_sample["eligibility_score"] = (100 * (1 - df_sample["risk_prob"])).round(1)

    # Approved Prediction
    df_sample["approved"] = df_sample["eligibility_score"] >= score_threshold

    # KPIs
    approved_count = int(df_sample["approved"].sum())
    rejected_count = int((~df_sample["approved"]).sum())

    # Expected defaults among approved (sum of probabilities)
    cumulative_default_risk = float(df_sample.loc[df_sample["approved"], "risk_prob"].sum())
    avg_risk_approved = float(df_sample.loc[df_sample["approved"], "risk_prob"].mean()) if approved_count > 0 else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Approved applicants", f"{approved_count:,}")
    k2.metric("Rejected applicants", f"{rejected_count:,}")
    k3.metric("Avg default risk (approved)", f"{avg_risk_approved:.2%}")

    #Table
    st.markdown("### Sample records (table)")
    # Show best first for readability
    df_table = df_sample.sort_values("eligibility_score", ascending=False).copy()

    cols_to_show = [c for c in (["SK_ID_CURR"] + selected_features + ["risk_prob", "eligibility_score", "approved"]) if c in df_table.columns]
    st.dataframe(df_table[cols_to_show].head(int(max_rows)), use_container_width=True)

    # Approved flag (ONLY based on threshold)
    df_sample["approved"] = df_sample["eligibility_score"] >= score_threshold

    # Histogram
    st.markdown("### Eligibility score distribution (sample)")

    if len(df_sample) == 0:
        st.info("No data in sample.")
    else:
        import numpy as np
        import matplotlib.pyplot as plt

        scores = df_sample["eligibility_score"].values

        # Bin scores into integers 0..100
        bins = np.arange(0, 102, 1)  # edges 0..101
        counts, edges = np.histogram(scores, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        fig, ax = plt.subplots(figsize=(10, 4))

        # 1) All bars (default)
        ax.bar(centers, counts, width=0.9, align="center")

        # 2) Overlay approved bars in red (score >= threshold)
        mask = centers >= score_threshold
        ax.bar(centers[mask], counts[mask], width=0.9, align="center", color="red")

        # 3) Threshold line
        ax.axvline(score_threshold, color="red", linewidth=2)

        ax.set_xlabel("Eligibility score (0–100)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 100)

        st.pyplot(fig, use_container_width=True)

        approved_count = int(df_sample["approved"].sum())
        total = int(len(df_sample))
        st.caption(f"Approved (score ≥ {score_threshold}): {approved_count} / {total} ({approved_count/total:.1%})")

# Customer View
with tab_customer:
    st.subheader("Your financial health & credit readiness (Customer View)")

    st.write("This view translates model signals into user-facing insights and a what-if simulator.")

    left, right = st.columns([1, 1])

    with left:
        # Simple user inputs mapped to model features
        income = st.slider("Estimated income (proxy)", 10000, 300000, 60000, 1000)
        credit = st.slider("Desired credit amount", 1000, 200000, 20000, 500)
        annuity = st.slider("Estimated annuity / monthly repayment proxy", 0, 50000, 8000, 500)
        goods_price = st.slider("Estimated goods price", 0, 250000, 20000, 500)

        days_employed = st.slider("Income stability proxy (days employed)", -20000, 0, -3000, 100)
        days_birth = st.slider("Age proxy (days birth)", -25000, -7000, -12000, 100)
        fam = st.slider("Family members", 1, 10, 3, 1)
        children = st.slider("Children", 0, 8, 1, 1)

        income_type = st.selectbox("Income regularity proxy (NAME_INCOME_TYPE)", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
        edu = st.selectbox("Education proxy (NAME_EDUCATION_TYPE)", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        occ = st.selectbox("Economic activity proxy (OCCUPATION_TYPE)", ["Laborers", "Sales staff", "Managers", "Core staff", "Drivers", "Cooking staff", "Security staff", "Cleaning staff", "Medicine staff", "Other"])
        fam_status = st.selectbox("Household structure proxy (NAME_FAMILY_STATUS)", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])

    # Build input row with the exact expected feature names
    input_row = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_BIRTH": days_birth,
        "CNT_FAM_MEMBERS": fam,
        "CNT_CHILDREN": children,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": edu,
        "OCCUPATION_TYPE": occ,
        "NAME_FAMILY_STATUS": fam_status
    }

    X_user = pd.DataFrame([input_row])[selected_features]
    risk_prob = float(clf.predict_proba(X_user)[0, 1])
    score = risk_to_score(risk_prob)

    with right:
        st.markdown("### My Snapshot")
        snap = pd.DataFrame({
            "Metric": ["Estimated default risk", "Eligibility score (0–100)"],
            "Value": [f"{risk_prob:.2f}", f"{score}"]
        })
        st.table(snap)

        st.markdown("### What-if simulator (income stability)")
        # vary stability and show score curve
        stability_range = np.linspace(-20000, 0, 25)
        scores = []
        for s in stability_range:
            tmp = input_row.copy()
            tmp["DAYS_EMPLOYED"] = float(s)
            p = float(clf.predict_proba(pd.DataFrame([tmp])[selected_features])[0, 1])
            scores.append(risk_to_score(p))
        chart_df = pd.DataFrame({"DAYS_EMPLOYED": stability_range, "Eligibility score": scores})
        st.line_chart(chart_df.set_index("DAYS_EMPLOYED"))

        st.markdown("### Tips to improve")
        tips = []
        if days_employed > -1500:
            tips.append("Increase income regularity: more consistent earning patterns improve your readiness score.")
        if credit > income * 1.0:
            tips.append("Start with a smaller loan amount to build repayment history and reduce risk.")
        if fam >= 5:
            tips.append("Higher household burden can reduce resilience—consider lower monthly repayment plans.")
        if not tips:
            tips.append("You’re in a good position—keep consistency and avoid sudden spending spikes.")
        for t in tips[:3]:
            st.write("• " + t)