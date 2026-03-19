# Boost - Smart Scoring for the Informal Sector

**Boost** is a Streamlit prototype for inclusive microcredit assessment. It combines a machine learning risk model with a customer-facing guidance experience designed for people with informal income, changing work patterns, or limited credit history.

Tagline: *Predicting potential, not just history.*

## Project Overview

The app is built around two complementary perspectives:

1. **Bank View** for portfolio pre-screening and threshold-based decision support.
2. **Customer Financial Health View** for individual score transparency, simulations, and guided next steps.

Beyond score prediction, the prototype includes **Micromate Boost AI**, a supportive assistant that helps applicants understand how to improve their loan readiness in simple, practical language.

## Main Features

### Bank View

- Adjustable eligibility threshold from 1 to 100.
- Portfolio KPIs for approved applicants, rejected applicants, and average default risk.
- Sample applicant table with predicted risk and eligibility score.
- Eligibility score distribution to explore approval trade-offs.

### Customer Financial Health View

- Individual eligibility score and risk band.
- Personalized feedback based on the current bank threshold.
- "What-if" simulation to test how profile changes affect the score.
- Action-oriented guidance for applicants who are not yet eligible.

### Micromate Boost AI

- Generates a personalized credit-readiness plan for the selected applicant.
- Supports multi-turn chat with conversation memory.
- Adapts guidance to the applicant profile and income type.
- Shares curated public resources when the user asks for examples, links, or success stories.
- Uses a warm, non-judgmental tone tailored to financially underserved users.

## AI Behavior and Context Design

The AI layer is intentionally split into different responsibilities:

- `get_ai_credit_plan()` uses internal Boost policy guidance to generate the initial recommendation plan.
- `summarize_boost_conversation()` also uses internal policy guidance to maintain a short internal memory of the conversation.
- `chat_with_boost_ai()` does **not** use internal policy documents directly during normal chat replies.

This separation is important because it prevents the assistant from confusing internal policy section titles with user-facing resources.

For the chat experience:

- Public, user-facing resources come only from curated links in `knowledge/success_ideas.json`.
- The chat prompt receives those curated links as visible shareable context.
- Internal policy titles such as `Boost policies section ...` are explicitly blocked from appearing in user-facing responses.

## Machine Learning Model

### Dataset

The model is trained on the **Home Credit Default Risk** dataset from Kaggle:
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

### Training

Training and experimentation are documented in [Model_Training.ipynb](/Users/mateobarreda/Development/ESADE/2_Term/Prototyping/Prototype/prototype_1/Model_Training.ipynb).

The notebook covers:

- Data cleaning
- Feature selection
- Encoding and preprocessing
- Logistic Regression training
- Evaluation using ROC-AUC

### Model Artifacts

The application loads:

- `Models/microcredit_model.joblib`
- `Models/model_metadata.joblib`

## Knowledge Files

The app uses a small knowledge layer to support the AI experience:

- `knowledge/boost_policies.md`: internal policy and tone guidance for planning and summarization.
- `knowledge/success_ideas.json`: curated public links and practical examples that can be shown to users.
- `knowledge/income_type_tips.json`: income-type-specific guidance used to personalize recommendations.

## Project Structure

```text
prototype_1/
├── assets/
│   └── boost_logo.svg
├── knowledge/
│   ├── boost_policies.md
│   ├── income_type_tips.json
│   └── success_ideas.json
├── Models/
│   ├── microcredit_model.joblib
│   └── model_metadata.joblib
├── Model_Training.ipynb
├── README.md
├── chat_utils.py
└── streamlitapp.py
```

## Requirements

Install the main dependencies with:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly cohere
```

You will also need:

- a `cohere.key` file in the project root containing your Cohere API key
- the training data file at `DATA/application_train.csv`

## Run the App

From the project root:

```bash
streamlit run streamlitapp.py
```

## Purpose and Disclaimer

This project is a prototype for inclusive finance exploration and product design. It is meant to demonstrate how machine learning and conversational AI can support financial guidance, but it is **not** a production-ready lending decision system and does not guarantee credit approval outcomes.
