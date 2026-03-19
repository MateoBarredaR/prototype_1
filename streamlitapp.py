import os
import json
import html
import joblib
import cohere
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =========================
# Streamlit config
# =========================
st.set_page_config(
    page_title="Boost: Smart scoring for the informal sector.",
    layout="wide"
)

st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #E8F0FE;
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 8px;
        color: #111827;
    }
    .ai-bubble {
        background-color: #EAF7EE;
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 8px;
        color: #111827;
    }
    .resource-card {
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        padding: 12px 14px;
        border-radius: 14px;
        margin-bottom: 10px;
    }
    .small-note {
        color: #6B7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Constants / paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

POLICIES_PATH = os.path.join(KNOWLEDGE_DIR, "boost_policies.md")
SUCCESS_IDEAS_PATH = os.path.join(KNOWLEDGE_DIR, "success_ideas.json")
INCOME_TYPE_TIPS_PATH = os.path.join(KNOWLEDGE_DIR, "income_type_tips.json")


# =========================
# Cohere API setup
# =========================
@st.cache_resource
def load_cohere_client():
    with open("cohere.key") as f:
        cohere_api_key = f.read().strip()
    return cohere.ClientV2(cohere_api_key)


cohere_client = load_cohere_client()


# =========================
# Model loading
# =========================
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "microcredit_model.joblib")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selected_features"]


clf, selected_features = load_model()


# =========================
# Session state
# =========================
DEFAULT_STATE = {
    "show_ai_chat": False,
    "boost_chat_history": [],
    "boost_chat_summary": "",
    "last_applicant_idx": None,
    "boost_ai_plan": None,
    "boost_chat_messages": [],
    "bank_score_threshold": 80,
    "suggested_links_for_chat": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================
# Data + knowledge loaders
# =========================
@st.cache_data
def load_data_sample(n=5000, seed=42):
    df = pd.read_csv(os.path.join("DATA", "application_train.csv"))
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df


@st.cache_data
def load_boost_policy_text():
    if os.path.exists(POLICIES_PATH):
        with open(POLICIES_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return """
# Boost policies

## Inclusion principles
Boost supports people with informal income, changing work patterns, or limited credit history.

## Refinancing guidance
- Lower monthly repayment pressure may improve repayment capacity.
- Smaller first loans may be easier to manage.
- Repeated income patterns can strengthen a profile.

## Language rules
- Never shame the user.
- Never call the user risky or unstable.
- Use respectful and encouraging language.
""".strip()


@st.cache_data
def load_boost_policy_documents():
    policy_text = load_boost_policy_text()
    chunks = [chunk.strip() for chunk in policy_text.split("\n\n") if chunk.strip()]

    documents = []
    for i, chunk in enumerate(chunks, start=1):
        documents.append(
            {
                "data": {
                    "title": f"Boost policies section {i}",
                    "text": chunk
                }
            }
        )
    return documents


@st.cache_data
def load_success_ideas():
    if os.path.exists(SUCCESS_IDEAS_PATH):
        with open(SUCCESS_IDEAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return [
        {
    "income_type": "Working",
    "title": "How to document informal or gig income",
    "url": "https://www.irs.gov/businesses/small-businesses-self-employed/manage-taxes-for-your-gig-work",
    "tag": "income_proof"
  },
  {
    "income_type": "Student",
    "title": "Building credit as a college student",
    "url": "https://www.investopedia.com/guide-to-student-credit-cards-4774771",
    "tag": "credit_building"
  },
  {
    "income_type": "Businessman",
    "title": "Small business financial management 101",
    "url": "https://www.sba.gov/business-guide/manage-your-business/manage-your-finances",
    "tag": "business_growth"
  }
    ]


@st.cache_data
def load_income_type_tips():
    if os.path.exists(INCOME_TYPE_TIPS_PATH):
        with open(INCOME_TYPE_TIPS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
  "Pensioner": {
    "label": "Pensioner",
    "ideas": [
      "Consider starting with a smaller loan amount to match your fixed income.",
      "Show regular pension deposits from the last 6 months to build trust.",
      "Keep monthly repayment pressure low to ensure financial comfort."
    ]
  },
  "Working": {
    "label": "Working",
    "ideas": [
      "Highlight stable, recurring income patterns over time.",
      "Document any overtime or bonuses as additional support.",
      "Starting with a short-term loan can quickly build your credit reputation."
    ]
  },
  "Commercial associate": {
    "label": "Commercial associate",
    "ideas": [
      "Show recurring sales or commissions to prove income flow.",
      "Document regular client payments or business contracts.",
      "A smaller first loan can demonstrate your capacity to manage business debt."
    ]
  },
  "State servant": {
    "label": "State servant",
    "ideas": [
      "Leverage your employment stability as a strong factor for approval.",
      "Ensure all official benefit documentation is up to date.",
      "Automatic payroll deductions can be a great way to guarantee on-time payments."
    ]
  },
  "Unemployed": {
    "label": "Unemployed",
    "ideas": [
      "Focus on showing any secondary income or government support.",
      "Starting with a micro-loan is the best way to prove reliability during transitions.",
      "Highlight your previous stable work history if available."
    ]
  },
  "Student": {
    "label": "Student",
    "ideas": [
      "Document any scholarships, grants, or part-time work income.",
      "A very small loan for educational tools is a great way to start your credit history.",
      "Consider a co-signer or proof of family support if possible."
    ]
  },
  "Businessman": {
    "label": "Businessman",
    "ideas": [
      "Provide tax returns or official business registration documents.",
      "Show a healthy cash flow in your business bank accounts.",
      "Use the loan for a specific, revenue-generating business purpose."
    ]
  },
  "Maternity leave": {
    "label": "Maternity leave",
    "ideas": [
      "Include documentation of maternity benefits and return-to-work plans.",
      "Focus on family income stability during this period.",
      "Small, flexible installments can help manage expenses during your leave."
    ]
  }
}


# =========================
# Helper functions
# =========================
def risk_to_score(risk_prob: float) -> float:
    return round(100 * (1 - risk_prob), 1)


def safe_text(value):
    if pd.isna(value):
        return "Unknown"
    return str(value)


def unique_keep_order(items):
    return list(dict.fromkeys(items))


def build_profile_summary(row, score, threshold):
    summary = {
        "score": round(float(score), 1),
        "threshold": round(float(threshold), 1),
        "income_type": safe_text(row.get("NAME_INCOME_TYPE", "Unknown")),
        "income_total": float(row.get("AMT_INCOME_TOTAL", 0) or 0),
        "credit_amount": float(row.get("AMT_CREDIT", 0) or 0),
        "annuity": float(row.get("AMT_ANNUITY", 0) or 0),
        "goods_price": float(row.get("AMT_GOODS_PRICE", 0) or 0),
        "days_employed": float(row.get("DAYS_EMPLOYED", 0) or 0),
        "days_birth": float(row.get("DAYS_BIRTH", 0) or 0),
        "family_members": float(row.get("CNT_FAM_MEMBERS", 0) or 0),
        "children": float(row.get("CNT_CHILDREN", 0) or 0),
    }

    derived_flags = []

    income = summary["income_total"]
    credit = summary["credit_amount"]
    annuity = summary["annuity"]

    if income > 0:
        summary["credit_to_income_ratio"] = round(credit / income, 3)
        summary["annuity_to_income_ratio"] = round(annuity / income, 3)
    else:
        summary["credit_to_income_ratio"] = None
        summary["annuity_to_income_ratio"] = None

    if income > 0 and credit > income:
        derived_flags.append("requested_credit_higher_than_income")

    if income > 0 and annuity > 0.20 * income:
        derived_flags.append("high_repayment_burden")

    if summary["days_employed"] > -1500:
        derived_flags.append("limited_income_stability")

    if summary["family_members"] >= 5:
        derived_flags.append("large_household_size")

    if summary["children"] >= 3:
        derived_flags.append("high_dependency_load")

    summary["derived_flags"] = derived_flags
    return summary


def get_profile_signals_for_display(profile_summary):
    signals = []

    income_type = profile_summary.get("income_type", "Unknown")
    signals.append(f"Current income type in profile: {income_type}")

    if profile_summary.get("credit_amount", 0) > 0:
        signals.append(f"Requested loan amount: {profile_summary['credit_amount']:.0f}")

    if profile_summary.get("income_total", 0) > 0:
        signals.append(f"Estimated income: {profile_summary['income_total']:.0f}")

    if profile_summary.get("annuity", 0) > 0:
        signals.append(f"Estimated repayment amount: {profile_summary['annuity']:.0f}")

    if profile_summary.get("family_members", 0) > 0:
        signals.append(f"Household size: {profile_summary['family_members']:.0f}")

    if profile_summary.get("children", 0) > 0:
        signals.append(f"Children: {profile_summary['children']:.0f}")

    flags = profile_summary.get("derived_flags", [])

    if "requested_credit_higher_than_income" in flags:
        signals.append("The requested amount may be difficult to support with the current income estimate.")

    if "high_repayment_burden" in flags:
        signals.append("The expected monthly burden may be a bit heavy right now.")

    if "limited_income_stability" in flags:
        signals.append("Showing more regular income patterns over time may help strengthen the profile.")

    if "large_household_size" in flags:
        signals.append("Household responsibilities may affect repayment flexibility.")

    if "high_dependency_load" in flags:
        signals.append("Family responsibilities may reduce available financial cushion.")

    return signals


def get_relevant_links(income_type, max_links=3):
    ideas = load_success_ideas()
    exact_matches = [x for x in ideas if x.get("income_type", "").lower() == str(income_type).lower()]
    generic_matches = [x for x in ideas if x.get("income_type", "").lower() == "generic"]
    return (exact_matches + generic_matches)[:max_links]


def get_income_type_ideas(income_type):
    tips_map = load_income_type_tips()
    entry = tips_map.get(income_type)
    if entry:
        return entry.get("ideas", [])
    return [
        "Start with a manageable loan amount.",
        "Show repeated income patterns if possible.",
        "Try to keep monthly repayment pressure low."
    ]


def build_resource_context(income_type, links):
    resource_lines = [f"Income type: {income_type}"]

    ideas = get_income_type_ideas(income_type)
    if ideas:
        resource_lines.append("Income-type ideas:")
        for idea in ideas:
            resource_lines.append(f"- {idea}")

    if links:
        resource_lines.append("Helpful resources:")
        for item in links:
            resource_lines.append(f"- {item['title']}: {item['url']}")

    return "\n".join(resource_lines)


def classify_user_message(user_message: str) -> str:
    msg = user_message.lower()

    if any(x in msg for x in ["you already repeated", "repeated", "same thing", "you are repeating", "that is repetitive"]):
        return "frustration_repetition"

    if any(x in msg for x in [
        "what do i do", "how do i do that", "how can i", "what can i do",
        "how do i", "what should i do", "what do i need", "where do i start"
    ]):
        return "action_request"

    if any(x in msg for x in [
        "link", "links", "resource", "resources", "example", "examples",
        "success example", "success examples", "website", "websites",
        "show me examples", "can you give me success examples"
    ]):
        return "links_request"

    if any(x in msg for x in ["side business", "not registered", "informal income", "cash income", "unregistered"]):
        return "new_income_detail"

    return "general"


def user_is_asking_for_links(user_message: str) -> bool:
    msg = user_message.lower()

    link_keywords = [
        "link", "links", "resource", "resources",
        "example", "examples", "success example", "success examples",
        "website", "websites", "where can i read", "show me examples"
    ]

    return any(keyword in msg for keyword in link_keywords)


def get_special_context(user_message: str) -> str:
    msg = user_message.lower()

    if "side business" in msg or "not registered" in msg or "informal income" in msg or "cash income" in msg or "unregistered" in msg:
        return """
The user says that an important part of their income comes from a side business that is not registered.
Focus on practical ways to demonstrate repeated income without formal registration.
Useful examples:
- bank transfer history
- screenshots of repeated payments
- sales logs
- invoices or receipts
- simple monthly income summaries
- proof of regular clients
Do not give legal advice.
Do not judge the user.
Focus on documentation, clarity, and practical next steps.
"""
    return ""


def get_ai_credit_plan(profile_summary):
    income_type = profile_summary.get("income_type", "Unknown")
    income_type_ideas = get_income_type_ideas(income_type)
    resource_context = build_resource_context(income_type, get_relevant_links(income_type))

    prompt = f"""
You are Micromate Boost AI, a friendly and supportive assistant inside Boost.

Boost is a financial inclusion product designed for people who may have informal income, changing work patterns, or limited credit history.

Your job is to explain the user's situation in a respectful, clear, and encouraging way.

Important rules:
- Use easy English.
- Sound warm, human, and supportive.
- Do NOT sound like a bank analyst.
- Do NOT sound cold, robotic, or overly brief.
- Do NOT use aggressive language.
- Do NOT say things like "you are risky", "you are unstable", or "you do not have a stable job".
- Recognize that informal income is still valid income.
- Use soft and respectful wording.
- Personalize the guidance with the applicant's income type only once.
- Do not repeat the income type in multiple sections unless necessary.
- Avoid saying the same idea in the summary, overview, and actions.
- Focus on what the user can improve next.
- Do NOT guarantee loan approval.
- Write complete, natural-sounding sentences.
- Return ONLY valid JSON.
- No markdown.
- No code fences.

Applicant profile:
{json.dumps(profile_summary, indent=2)}

Helpful context:
{resource_context}

Example income-type ideas:
{json.dumps(income_type_ideas, indent=2)}

Return exactly this JSON structure:
{{
  "friendly_summary": "A short but warm explanation of the user's current situation in 2-4 sentences.",
  "profile_overview": {{
    "strength": "One positive thing about the profile",
    "main_concern": "Main thing currently holding the score back, phrased gently",
    "next_best_step": "Most useful next action"
  }},
  "income_type_message": "One short sentence that respectfully connects the income type to the advice.",
  "recommended_actions": [
    {{
      "action": "Short action title",
      "explanation": "A natural and supportive explanation in easy English",
      "priority": "high"
    }},
    {{
      "action": "Short action title",
      "explanation": "A natural and supportive explanation in easy English",
      "priority": "medium"
    }},
    {{
      "action": "Short action title",
      "explanation": "A natural and supportive explanation in easy English",
      "priority": "medium"
    }}
  ],
  "chat_intro": "A warm sentence inviting the user to ask questions to the AI assistant"
}}
"""

    response = cohere_client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        documents=load_boost_policy_documents(),
        temperature=0.55
    )

    text = response.message.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "friendly_summary": "Your profile already shows some positive signals, and that is a good starting point. At the moment, the requested amount and the expected repayment effort may be a bit heavy for your current financial picture. A few adjustments could help make your application stronger.",
            "profile_overview": {
                "strength": "You already have enough financial information for Boost to evaluate your profile.",
                "main_concern": "The loan may be a little too large for your current repayment capacity.",
                "next_best_step": "Starting with a smaller amount or reducing the repayment burden could help first."
            },
            "income_type_message": f"I can see your current income type is {income_type}, so some of the suggestions can be adapted to that situation.",
            "recommended_actions": [
                {
                    "action": "Start with a smaller loan",
                    "explanation": "A smaller first loan may feel more manageable and can make your profile look stronger for a lender.",
                    "priority": "high"
                },
                {
                    "action": "Show regular income patterns",
                    "explanation": "Any proof of repeated income, even from informal work or side activities, can help make your profile clearer and more trustworthy.",
                    "priority": "medium"
                },
                {
                    "action": "Make monthly payments lighter",
                    "explanation": "Lower monthly pressure can help show that the loan fits more comfortably into your current finances.",
                    "priority": "medium"
                }
            ],
            "chat_intro": "If you want, you can ask me more specific questions and I’ll help you think through the next step."
        }


def summarize_boost_conversation(profile_summary, previous_summary, chat_history):
    recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
    formatted_history = "\n".join([f"{speaker}: {message}" for speaker, message in recent_messages])

    prompt = f"""
You are Micromate Boost AI.

You are updating a short internal conversation summary for a user support chat inside Boost, a bank that supports people with informal income, non-traditional work patterns, or limited credit history.

Important rules:
- Be respectful and non-judgmental.
- Do not use stigmatizing language.
- Do not say the person is unreliable or unstable.
- Use neutral wording such as "income pattern", "repayment capacity", or "financial flexibility".
- Keep the summary short and cumulative.
- Preserve important user goals, personal details they shared, and the most relevant advice already given.
- Highlight new facts shared by the user, especially about informal income or side businesses.
- Return only plain text.

Applicant profile:
{json.dumps(profile_summary, indent=2)}

Previous summary:
{previous_summary}

Recent conversation:
{formatted_history}

Write an updated summary in 4-6 short sentences.
"""

    response = cohere_client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        documents=load_boost_policy_documents(),
        temperature=0.25
    )
    return response.message.content[0].text.strip()


def chat_with_boost_ai(profile_summary, conversation_summary, chat_messages, resource_context, user_message):
    income_type = profile_summary.get("income_type", "Unknown")
    message_type = classify_user_message(user_message)
    special_context = get_special_context(user_message)

    system_prompt = f"""
You are Micromate Boost AI, a friendly and supportive assistant inside Boost.

Boost is a financial inclusion product designed for people who may have informal income, changing work patterns, or limited credit history.
Your role is to help users understand how to improve their loan readiness in a respectful, practical, and encouraging way.

Important rules:
- Use very simple English.
- Sound warm, human, and supportive.
- Never sound cold, abrupt, or robotic.
- Never shame the user.
- Never say things like "you do not have a stable job" or "your profile is bad".
- Recognize that informal work is still real work.
- Do not promise approval.
- Be specific and helpful.
- Prefer concrete actions over generic advice.
- Do not repeat the same advice if it was already given.
- Always respond first to NEW information shared by the user.
- If the user says the answer is repetitive, briefly acknowledge it and then give a more specific answer.
- If the user asks "what do I do?" or "how do I do that?", give step-by-step practical guidance.
- If the user asks for examples, links, resources, or websites, clearly say that you are also sharing useful resources below.
- When resources are available, briefly refer to them in your answer.
- Do not invent URLs.
- Do not make up organizations, articles, or websites.
- When curated links are available, refer to them as real examples the user can open below.
- If the user asks for success stories, give a short practical answer first, and then mention that relevant links and examples are shown below.
- Avoid mentioning the income type in every answer.
- Mention the income type only when it truly adds value.
- Use complete, natural-sounding sentences.
- Try to sound like a thoughtful human coach, not a compliance bot.
- A good answer is usually 4-8 sentences unless the user asks for something shorter.

Applicant profile:
{json.dumps(profile_summary, indent=2)}

Conversation summary so far:
{conversation_summary}

Helpful resource context:
{resource_context}

Current income type:
{income_type}

Current user message type:
{message_type}

Extra context:
{special_context}
"""

    messages = [{"role": "system", "content": system_prompt}] + chat_messages

    response = cohere_client.chat(
        model="command-a-03-2025",
        messages=messages,
        documents=load_boost_policy_documents(),
        temperature=0.55
    )

    return response.message.content[0].text.strip()


def render_chat_history():
    for speaker, message in st.session_state["boost_chat_history"]:
        role = "user" if speaker == "You" else "assistant"
        avatar = "🧑" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):
            css_class = "user-bubble" if role == "user" else "ai-bubble"
            st.markdown(
                f'<div class="{css_class}">{html.escape(message)}</div>',
                unsafe_allow_html=True
            )


# =========================
# Header
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/boost_logo.svg", width=1800)

st.subheader("Predicting potential, not just history")
st.markdown(
    """
    Boost is a microcredit risk assessment tool designed for the informal sector. 
    It combines machine learning with financial health insights to empower both lenders and borrowers.
    """
)
st.caption("Two perspectives: Bank decisioning and Customer financial health.")

tab_bank, tab_customer = st.tabs(["Bank View", "Customer Financial Health View"])


# =========================
# Bank View
# =========================
with tab_bank:
    st.subheader("Portfolio pre-screening (Bank View)")

    col1, col2, col3 = st.columns(3)
    with col1:
        score_threshold = st.slider("Eligibility score threshold", 1, 100, 80, 1)
        st.session_state["bank_score_threshold"] = score_threshold

    with col2:
        sample_size = st.selectbox("Sample size", [1000, 3000, 5000, 10000], index=2)
    with col3:
        max_rows = st.number_input("Rows to display (table)", value=25, min_value=5, max_value=10000)

    df_sample = load_data_sample(n=sample_size)

    X_bank = df_sample[selected_features].copy()
    risk_probs = clf.predict_proba(X_bank)[:, 1]
    df_sample["risk_prob"] = risk_probs
    df_sample["eligibility_score"] = (100 * (1 - df_sample["risk_prob"])).round(1)
    df_sample["approved"] = df_sample["eligibility_score"] >= score_threshold

    approved_count = int(df_sample["approved"].sum())
    rejected_count = int((~df_sample["approved"]).sum())
    avg_risk_approved = float(df_sample.loc[df_sample["approved"], "risk_prob"].mean()) if approved_count > 0 else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Approved applicants", f"{approved_count:,}")
    k2.metric("Rejected applicants", f"{rejected_count:,}")
    k3.metric("Avg default risk (approved)", f"{avg_risk_approved:.2%}")

    st.markdown("### Sample records (table)")
    df_table = df_sample.sort_values("eligibility_score", ascending=False).copy()
    cols_to_show = [
        c for c in (["SK_ID_CURR"] + selected_features + ["risk_prob", "eligibility_score", "approved"])
        if c in df_table.columns
    ]
    cols_to_show = unique_keep_order(cols_to_show)
    st.dataframe(df_table[cols_to_show].head(int(max_rows)), use_container_width=True)

    st.markdown("### Eligibility score distribution (sample)")

    if len(df_sample) == 0:
        st.info("No data in sample.")
    else:
        scores = df_sample["eligibility_score"].values
        bins = np.arange(0, 102, 1)
        counts, edges = np.histogram(scores, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        base_color = "#C9CCD2"
        approved_color = "#1DA1F2"
        threshold_color = "#E5E7EB"

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#505C8056")
        ax.set_facecolor("#0B0F19")

        ax.bar(centers, counts, width=0.9, align="center", color=base_color)
        mask = centers >= score_threshold
        ax.bar(centers[mask], counts[mask], width=0.9, align="center", color=approved_color)

        ax.axvline(score_threshold, color=threshold_color, linewidth=2)
        ax.set_xlabel("Eligibility score (0–100)", color="white")
        ax.set_ylabel("Applicants", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig, use_container_width=True)

        total = int(len(df_sample))
        st.caption(f"Approved (score ≥ {score_threshold}): {approved_count} / {total} ({approved_count/total:.1%})")


# =========================
# Customer View
# =========================
with tab_customer:
    st.subheader("Customer View — Eligibility score")
    st.caption("Choose an applicant from the dataset and view their microcredit readiness score.")

    cust_account_tab, cust_sim_tab = st.tabs(["👤 Account", "🧪 Simulation"])

    with cust_account_tab:
        df_cust = load_data_sample(n=100)
        idx = st.selectbox("YOUR Account Number", list(range(len(df_cust))), key="account_idx")

        if st.session_state["last_applicant_idx"] != idx:
            st.session_state["show_ai_chat"] = False
            st.session_state["boost_chat_history"] = []
            st.session_state["boost_chat_summary"] = ""
            st.session_state["boost_ai_plan"] = None
            st.session_state["boost_chat_messages"] = []
            st.session_state["suggested_links_for_chat"] = []
            st.session_state["last_applicant_idx"] = idx

        row = df_cust.iloc[idx]

        X_one = pd.DataFrame([row[selected_features].to_dict()])[selected_features]
        risk_prob = float(clf.predict_proba(X_one)[0, 1])
        score = float(100 * (1 - risk_prob))

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

            bank_threshold = st.session_state.get("bank_score_threshold", 80)
            st.caption(f"Current bank eligibility threshold: **{bank_threshold}**")

            if score >= bank_threshold:
                st.success("You are eligible for a loan!")
                if st.button("Apply for a microloan", key="apply_loan_btn"):
                    st.balloons()
                    st.info("Application submitted (demo). Next: identity verification + repayment plan selection.")
            else:
                gap = bank_threshold - score
                st.warning(f"You are close — improving a few areas could help you gain the **{gap:.1f}** points you need.")

                st.markdown("### How would you like to improve your score?")
                st.caption("Recommendations on how to improve your score.")

                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("Open Simulation Tool", key="open_simulation_btn", use_container_width=True):
                        st.info("Go to the 🧪 Simulation tab above to test how changing your inputs affects your score.")

                with btn_col2:
                    if st.button("Chat with Micromate Boost AI", key="open_ai_chat_btn", use_container_width=True):
                        st.session_state["show_ai_chat"] = True

                if st.session_state.get("show_ai_chat", False):
                    profile_summary = build_profile_summary(row, score, bank_threshold)
                    income_type = profile_summary.get("income_type", "Unknown")
                    relevant_links = get_relevant_links(income_type)
                    resource_context = build_resource_context(income_type, relevant_links)

                    if st.session_state["boost_ai_plan"] is None:
                        with st.spinner("Preparing your personalized recommendations..."):
                            st.session_state["boost_ai_plan"] = get_ai_credit_plan(profile_summary)

                    ai_output = st.session_state["boost_ai_plan"]

                    st.markdown("### Your profile at a glance")
                    st.write(ai_output["friendly_summary"])
                    st.caption(f"Current income type in profile: **{income_type}**")
                    st.info(ai_output.get("income_type_message", f"I can see your current income type is {income_type}. Some ideas can be adapted to that situation."))

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**What is already helping you**")
                        st.write(ai_output["profile_overview"]["strength"])
                    with c2:
                        st.markdown("**What is holding you back**")
                        st.write(ai_output["profile_overview"]["main_concern"])
                    with c3:
                        st.markdown("**Best next step**")
                        st.write(ai_output["profile_overview"]["next_best_step"])

                    st.markdown("### Suggested actions")
                    for rec in ai_output["recommended_actions"]:
                        st.markdown(f"**{rec['action']}**")
                        st.write(rec["explanation"])
                        st.caption(f"Priority: {rec['priority']}")

                    income_type_ideas = get_income_type_ideas(income_type)
                    if income_type_ideas:
                        st.markdown(f"### Ideas for your income type: {income_type}")
                        for idea in income_type_ideas:
                            st.write(f"- {idea}")

                    if relevant_links:
                        st.markdown("### Helpful ideas and examples")
                        for item in relevant_links:
                            st.markdown(
                                f"""
                                <div class="resource-card">
                                    <strong>{html.escape(item['title'])}</strong><br>
                                    <span class="small-note">{html.escape(item.get('tag', 'resource').title())}</span><br>
                                    <a href="{html.escape(item['url'])}" target="_blank">Open resource</a>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.info(ai_output["chat_intro"])

                    st.markdown("### Chat with Boost AI")
                    render_chat_history()

                    if st.session_state["suggested_links_for_chat"]:
                        st.markdown("### Suggested resources")
                        for item in st.session_state["suggested_links_for_chat"]:
                            st.markdown(
                                f"""
                                <div class="resource-card">
                                    <strong>{html.escape(item['title'])}</strong><br>
                                    <span class="small-note">{html.escape(item.get('tag', 'resource').title())}</span><br>
                                    <a href="{html.escape(item['url'])}" target="_blank">Open resource</a>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    user_message = st.chat_input("Ask Boost AI a question")

                    if user_message and user_message.strip():
                        user_message = user_message.strip()

                        st.session_state["boost_chat_history"].append(("You", user_message))
                        st.session_state["boost_chat_messages"].append({
                            "role": "user",
                            "content": user_message
                        })

                        with st.spinner("Boost AI is thinking..."):
                            reply = chat_with_boost_ai(
                                profile_summary=profile_summary,
                                conversation_summary=st.session_state["boost_chat_summary"],
                                chat_messages=st.session_state["boost_chat_messages"],
                                resource_context=resource_context,
                                user_message=user_message
                            )

                        st.session_state["boost_chat_history"].append(("Boost AI", reply))
                        st.session_state["boost_chat_messages"].append({
                            "role": "assistant",
                            "content": reply
                        })

                        if user_is_asking_for_links(user_message):
                            st.session_state["suggested_links_for_chat"] = relevant_links
                        else:
                            st.session_state["suggested_links_for_chat"] = []

                        with st.spinner("Updating conversation memory..."):
                            st.session_state["boost_chat_summary"] = summarize_boost_conversation(
                                profile_summary=profile_summary,
                                previous_summary=st.session_state["boost_chat_summary"],
                                chat_history=st.session_state["boost_chat_history"]
                            )

                        st.rerun()

                    with st.expander("What information is Boost AI using?"):
                        st.write("Boost AI is using a few simple signals from your profile, such as:")
                        for signal in get_profile_signals_for_display(profile_summary):
                            st.write(f"- {signal}")
                        st.write("It also uses Boost policy guidance and curated support resources.")

        with colB:
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
                        {"range": [0, 50], "color": "#DB3437"},
                        {"range": [50, 75], "color": "#F6C609"},
                        {"range": [75, 100], "color": "#39B867"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.85,
                        "value": score
                    }
                },
                title={"text": "Microcredit eligibility ring"}
            ))

            fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show selected applicant inputs (proxies)"):
                show_cols = [c for c in (["SK_ID_CURR"] + selected_features + ["NAME_INCOME_TYPE"]) if c in df_cust.columns]
                show_cols = unique_keep_order(show_cols)
                st.dataframe(row[show_cols].to_frame().T, use_container_width=True)

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
                base_input[c] = st.selectbox(
                    f"{c} (proxy)",
                    options,
                    index=options.index(current),
                    key=f"sim_{c}"
                )

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