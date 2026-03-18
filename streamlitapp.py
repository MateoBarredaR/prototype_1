import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import json
import cohere

#Cohere API setup (for future use in tips generation)
@st.cache_resource
def load_cohere_client():
    with open("cohere.key") as f:
        COHERE_API_KEY = f.read().strip()
    return cohere.ClientV2(COHERE_API_KEY)

cohere_client = load_cohere_client()

# Page logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/boost_logo.svg", width=1800)
st.set_page_config(page_title="Boost: Smart scoring for the informal sector.", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "microcredit_model.joblib")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selected_features"]

clf, selected_features = load_model()

def risk_to_score(risk_prob: float) -> float:
    return round(100 * (1 - risk_prob), 1)

st.subheader("Predicting potential, not just history")
st.markdown(
    """
    Boost is a microcredit risk assessment tool designed for the informal sector. 
    It combines machine learning with financial health insights to empower both lenders and borrowers.
    """
)
st.caption("Two perspectives: Bank decisioning and Customer financial health.")

tab_bank, tab_customer = st.tabs(["Bank View", "Customer Financial Health View"])

# Sampling Data
@st.cache_data
def load_data_sample(n=5000, seed=42):
    df = pd.read_csv(os.path.join("DATA", "application_train.csv"))
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df

### Bank View
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

        scores = df_sample["eligibility_score"].values

        # Bin scores into integers 0..100
        bins = np.arange(0, 102, 1)  # edges 0..101
        counts, edges = np.histogram(scores, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        # Colors
        base_color = "#C9CCD2"
        approved_color = "#1DA1F2"
        threshold_color = "#E5E7EB"

        fig, ax = plt.subplots(figsize=(10, 4))

        # Dark background
        fig.patch.set_facecolor("#505C8056")
        ax.set_facecolor("#0B0F19")

        # 1) All bars (default)
        ax.bar(centers, counts, width=0.9, align="center", color=base_color)

        # 2) Overlay approved bars in red (score >= threshold)
        mask = centers >= score_threshold
        ax.bar(centers[mask], counts[mask], width=0.9, align="center", color=approved_color)

        # 3) Threshold line
        ax.axvline(score_threshold, color=threshold_color, linewidth=2)

        # Labels (white)
        ax.set_xlabel("Eligibility score (0–100)", color="white")
        ax.set_ylabel("Applicants", color="white")

        # Ticks (white)
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        # Remove spines (clean look)
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig, use_container_width=True)

        approved_count = int(df_sample["approved"].sum())
        total = int(len(df_sample))
        st.caption(f"Approved (score ≥ {score_threshold}): {approved_count} / {total} ({approved_count/total:.1%})")

### Customer View

with tab_customer:
    st.subheader("Customer View — Eligibility score")
    st.caption("Choose an applicant from the dataset and view their microcredit readiness score.")

    # Sub-tabs SOLO dentro de Customer
    cust_account_tab, cust_sim_tab = st.tabs(["👤 Account", "🧪 Simulation"])

   
    ## Account tab

    with cust_account_tab:
        df_cust = load_data_sample(n=100)
        idx = st.selectbox("YOUR Account Number", list(range(len(df_cust))))
        row = df_cust.iloc[idx]

        # Predict for selected applicant
        X_one = pd.DataFrame([row[selected_features].to_dict()])[selected_features]
        risk_prob = float(clf.predict_proba(X_one)[0, 1])
        score = float((100 * (1 - risk_prob)))

        # Decide color band
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
                if st.button("Apply for a microloan", key="apply_loan_btn"):
                    st.balloons()
                    st.info("Application submitted (demo). Next: identity verification + repayment plan selection.")
            else:
                gap = bank_threshold - score
                st.warning(f"Almost there — you are **{gap:.1f}** points away.")
                if st.button(
                    "Almost there: click here to know how you can improve your score",
                    key="improve_score_btn"
                ):
                    st.markdown("#### Boost your score: Tips & Simulation (Check our Simulation tab)")
                    st.caption("AI recommendations based on your profile (comming soon):")
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
            # Circular ring gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "", "font": {"size": 51}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#DB3437"},    # red
                        {"range": [50, 75], "color": "#F6C609"},   # yellow
                        {"range": [75, 100], "color": "#39B867"}   # green
                    ],
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

            with st.expander("Show selected applicant inputs (proxies)"):
                show_cols = [c for c in (["SK_ID_CURR"] + selected_features) if c in df_cust.columns]
                st.dataframe(pd.DataFrame([row[show_cols]]), use_container_width=True)


    ## Simulation tab 

    with cust_sim_tab:
        st.subheader("Simulation — explore how inputs affect your score")
        st.caption("This is a sandbox. Changes are not saved. Adjust inputs to see the eligibility score update instantly.")

        df_sim = load_data_sample(n=200)
        base_idx = st.selectbox("Start from an example applicant", list(range(len(df_sim))), index=0, key="sim_base_idx")
        base_row = df_sim.iloc[base_idx]

        base_input = {f: base_row[f] for f in selected_features}

        left, right = st.columns([1, 2])

        with left:
            st.markdown("### Adjust inputs")

            num_feats = [c for c in selected_features if pd.api.types.is_numeric_dtype(df_sim[c])]
            cat_feats = [c for c in selected_features if c not in num_feats]

            def bounded_slider(name, value, min_v, max_v, step):
                if value < min_v:
                    value = min_v
                if value > max_v:
                    value = max_v
                return st.slider(name, min_v, max_v, value, step)

            if "AMT_INCOME_TOTAL" in num_feats:
                base_input["AMT_INCOME_TOTAL"] = bounded_slider(
                    "Estimated income (AMT_INCOME_TOTAL)",
                    int(base_input["AMT_INCOME_TOTAL"]),
                    10_000, 300_000, 1_000
                )

            if "AMT_CREDIT" in num_feats:
                base_input["AMT_CREDIT"] = bounded_slider(
                    "Desired credit amount (AMT_CREDIT)",
                    int(base_input["AMT_CREDIT"]),
                    500, 200_000, 500
                )

            if "AMT_ANNUITY" in num_feats:
                base_input["AMT_ANNUITY"] = bounded_slider(
                    "Repayment burden proxy (AMT_ANNUITY)",
                    int(base_input["AMT_ANNUITY"]),
                    0, 50_000, 500
                )

            if "AMT_GOODS_PRICE" in num_feats:
                base_input["AMT_GOODS_PRICE"] = bounded_slider(
                    "Goods / investment value (AMT_GOODS_PRICE)",
                    int(base_input["AMT_GOODS_PRICE"]) if pd.notna(base_input["AMT_GOODS_PRICE"]) else 0,
                    0, 250_000, 500
                )

            if "DAYS_EMPLOYED" in num_feats:
                base_input["DAYS_EMPLOYED"] = bounded_slider(
                    "Income stability proxy (DAYS_EMPLOYED)",
                    int(base_input["DAYS_EMPLOYED"]) if pd.notna(base_input["DAYS_EMPLOYED"]) else -3000,
                    -20_000, 0, 100
                )

            if "DAYS_BIRTH" in num_feats:
                base_input["DAYS_BIRTH"] = bounded_slider(
                    "Age proxy (DAYS_BIRTH)",
                    int(base_input["DAYS_BIRTH"]) if pd.notna(base_input["DAYS_BIRTH"]) else -12000,
                    -25_000, -7_000, 100
                )

            if "CNT_FAM_MEMBERS" in num_feats:
                base_input["CNT_FAM_MEMBERS"] = bounded_slider(
                    "Household size (CNT_FAM_MEMBERS)",
                    int(base_input["CNT_FAM_MEMBERS"]) if pd.notna(base_input["CNT_FAM_MEMBERS"]) else 3,
                    1, 10, 1
                )

            if "CNT_CHILDREN" in num_feats:
                base_input["CNT_CHILDREN"] = bounded_slider(
                    "Children (CNT_CHILDREN)",
                    int(base_input["CNT_CHILDREN"]) if pd.notna(base_input["CNT_CHILDREN"]) else 1,
                    0, 8, 1
                )

            for c in cat_feats:
                options = sorted([x for x in df_sim[c].dropna().unique().tolist()])
                if not options:
                    continue
                current = base_input[c]
                if current not in options:
                    current = options[0]
                base_input[c] = st.selectbox(f"{c} (proxy)", options, index=options.index(current), key=f"sim_{c}")

        X_sim = pd.DataFrame([base_input])[selected_features]
        sim_risk = float(clf.predict_proba(X_sim)[0, 1])
        sim_score = float(100 * (1 - sim_risk))

        if sim_score >= 75:
            sim_band = "High"
        elif sim_score >= 50:
            sim_band = "Medium"
        else:
            sim_band = "Low"

        with right:
            st.markdown("### Result (updates live)")
            k1, k2, k3 = st.columns(3)
            k1.metric("Eligibility score (0–100)", f"{sim_score:.1f}")
            k2.metric("Estimated default risk", f"{sim_risk:.2%}")
            k3.metric("Band", sim_band)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_score,
                number={"font": {"size": 51}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#DB3437"},
                        {"range": [50, 75], "color": "#F6C609"},
                        {"range": [75, 100], "color": "#39B867"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.85,
                        "value": sim_score
                    }
                },
                title={"text": "Eligibility score (simulation)"}
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("See simulated input (debug)"):
                st.dataframe(X_sim, use_container_width=True)