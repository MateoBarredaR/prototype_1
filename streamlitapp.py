import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        # Global variable to share threshold with customer view
        st.session_state["bank_score_threshold"] = score_threshold

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
    st.subheader("Customer View — Eligibility score")
    st.caption("Choose an applicant from the dataset and view their microcredit readiness score.")

    # --------- Load a sample to pick an applicant ----------

    df_cust = load_data_sample(n=100)
    idx = st.selectbox("YOUR Account Number", list(range(len(df_cust))))
    row = df_cust.iloc[idx] 

    # --------- Predict for selected applicant ----------
    X_one = pd.DataFrame([row[selected_features].to_dict()])[selected_features]
    risk_prob = float(clf.predict_proba(X_one)[0, 1])
    score = float((100 * (1 - risk_prob)))

    # Decide color band
    # (Used for text + gauge steps)
    if score >= 75:
        band = "High"
    elif score >= 50:
        band = "Medium"
    else:
        band = "Low"

    colA, colB = st.columns([1, 2])

    with colA:
        k1, k2 = st.columns(2)
        k1.metric("Eligibility score (0–100)", f"{score:.1f}")
        k2.metric("Band", band)

        # Read threshold from session state (set by bank view) or default to 80
        bank_threshold = st.session_state.get("bank_score_threshold", 80)

        st.caption(f"Current bank eligibility threshold: **{bank_threshold}**")

        # Dynamic button and message based on eligibility
        if score >= bank_threshold:
            st.success("You are eligible for a loan!")
            if st.button("Apply for a microloan"):
                st.balloons()
                st.info("Application submitted (demo). Next: identity verification + repayment plan selection.")
        else:
            gap = bank_threshold - score
            st.warning(f"Almost there — you are **{gap:.1f}** points away.")
            if st.button("Almost there: click here to know how you can improve your score"):
                st.markdown("#### How to improve your score (personalized tips)")
                tips = []
                # simple tips
                if row.get("DAYS_EMPLOYED", -999999) > -1500:
                    tips.append("- Increase income regularity over time (more stable patterns improve readiness).")
                if row.get("AMT_CREDIT", 0) > row.get("AMT_INCOME_TOTAL", 1):
                    tips.append("- Consider requesting a smaller amount first to build a positive repayment history.")
                if row.get("CNT_FAM_MEMBERS", 0) >= 5:
                    tips.append("- A larger household can reduce financial resilience—smaller repayments may help.")
                if row.get("AMT_ANNUITY", 0) > 0 and row.get("AMT_ANNUITY", 0) > 0.2 * row.get("AMT_INCOME_TOTAL", 0):
                    tips.append("- Reduce repayment burden (lower monthly payments can reduce risk).")

                if not tips:
                    tips = ["- Keep consistency and avoid sudden spending spikes."]

                st.write("\n".join(tips))

    with colB:
        # --------- Circular ring gauge ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 51}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
                "bar": {"color": "rgba(0,0,0,0)"},  # hide the default bar
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#EF4444"},    # red
                    {"range": [50, 75], "color": "#F59E0B"},   # yellow
                    {"range": [75, 100], "color": "#22C55E"}   # green
                ],
                # A thin needle/marker at current score
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.85,
                    "value": score
                }
            },
            title={"text": "Microcredit eligibility ring"}
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=60, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Optional: show selected applicant details (light)
        with st.expander("Show selected applicant inputs (proxies)"):
            show_cols = [c for c in (["SK_ID_CURR"] + selected_features) if c in df_cust.columns]
            st.dataframe(pd.DataFrame([row[show_cols]]), use_container_width=True)