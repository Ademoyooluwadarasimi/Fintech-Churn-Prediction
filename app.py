import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------------------------------------------
# 1. APP CONFIGURATION AND LOAD MODEL FILES
# ------------------------------------------------------------
st.set_page_config(page_title="Fintech Churn Predictor", page_icon="📉", layout="wide")

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    """Load the three files exported from Colab (same names as the notebook)."""
    log_model = joblib.load(BASE_DIR / "log_model.joblib")                # the trained LogisticRegression
    scaler = joblib.load(BASE_DIR / "scaler.joblib")                      # the fitted StandardScaler
    feature_columns = joblib.load(BASE_DIR / "feature_columns.joblib")    # list(X_train.columns)
    return log_model, scaler, feature_columns


try:
    log_model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Could not find a model file: {e}. Put log_model.joblib, scaler.joblib and "
        "feature_columns.joblib in the same folder as app.py."
    )
    st.stop()



@st.cache_resource
def load_ranges():
    """Optional: min/max of each raw feature in the training data (for the guardrail)."""
    path = BASE_DIR / "feature_ranges.joblib"
    return joblib.load(path) if path.exists() else None


feature_ranges = load_ranges()

# Risk bands (same cut-offs used in the notebook)
LOW_CUTOFF = 0.35
HIGH_CUTOFF = 0.60

# Readable names for the "why this score" table
FRIENDLY_NAMES = {
    "age": "Age",
    "monthly_txn": "Monthly transactions",
    "avg_txn_amount": "Average transaction amount",
    "failed_txn_rate": "Failed transaction rate",
    "support_tickets": "Support tickets",
    "days_since_last_login": "Days since last login",
    "uses_mobile_app": "Uses mobile app",
    "app_missing": "Mobile app status unknown",
    "tenure_months": "Tenure (months)",
    "monthly_value_log": "Monthly value (log)",
    "country_ghana": "Country: Ghana",
    "country_kenya": "Country: Kenya",
    "country_nigeria": "Country: Nigeria",
    "country_united kingdom": "Country: United Kingdom",
    "plan_Premium": "Plan: Premium",
    "plan_Standard": "Plan: Standard",
}

# ------------------------------------------------------------
# 2. PAGE HEADER
# ------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption(
    "Enter a customer's details to estimate how likely they are to leave. "
    "Model: logistic regression trained on customer behaviour data."
)

# ------------------------------------------------------------
# 3. INPUT FORM
# ------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)
        country = st.selectbox("Country", ["Nigeria", "Ghana", "Kenya", "United Kingdom", "Canada"])
        plan = st.selectbox("Plan", ["Basic", "Standard", "Premium"])
        uses_app_label = st.radio("Uses the mobile app?", ["Yes", "No"], horizontal=True)

    with c2:
        tenure_months = st.number_input("Tenure (months with us)", min_value=0.0, max_value=120.0,
                                        value=25.0, step=1.0)
        monthly_txn = st.number_input("Transactions per month", min_value=0, max_value=500,
                                      value=19, step=1)
        avg_txn_amount = st.number_input("Average transaction amount (₦)", min_value=0.0,
                                         value=26800.0, step=500.0)

    with c3:
        failed_txn_pct = st.slider("Failed transactions (%)", min_value=0.0, max_value=100.0,
                                   value=9.0, step=0.5)
        support_tickets = st.number_input("Support tickets raised", min_value=0, max_value=50,
                                          value=1, step=1)
        days_since_last_login = st.number_input("Days since last login", min_value=0, max_value=365,
                                                value=10, step=1)

    submitted = st.form_submit_button("Predict churn risk")

# ------------------------------------------------------------
# 4. BUILD THE FEATURE ROW (same 16 columns, same order as training)
# ------------------------------------------------------------
def build_feature_row():
    row = {col: 0.0 for col in feature_columns}   # start with all zeros

    # numeric features
    row["age"] = float(age)
    row["monthly_txn"] = float(monthly_txn)
    row["avg_txn_amount"] = float(avg_txn_amount)
    row["failed_txn_rate"] = failed_txn_pct / 100.0          # model expects a 0-1 rate
    row["support_tickets"] = float(support_tickets)
    row["days_since_last_login"] = float(days_since_last_login)
    row["tenure_months"] = float(tenure_months)

    # engineered feature: log(1 + monthly_txn * avg_txn_amount)
    row["monthly_value_log"] = float(np.log1p(monthly_txn * avg_txn_amount))

    # binary features (1 / 0)
    row["uses_mobile_app"] = 1.0 if uses_app_label == "Yes" else 0.0
    row["app_missing"] = 0.0                                  # the user told us, so it is not missing

    # one-hot country (Canada is the baseline -> all country columns stay 0)
    country_col = f"country_{country.lower()}"
    if country_col in row:
        row[country_col] = 1.0

    # one-hot plan (Basic is the baseline -> all plan columns stay 0)
    plan_col = f"plan_{plan}"
    if plan_col in row:
        row[plan_col] = 1.0

    # DataFrame with the exact column order used in training
    return pd.DataFrame([row])[list(feature_columns)]


# ------------------------------------------------------------
# 5. PREDICT AND SHOW RESULTS
# ------------------------------------------------------------
if submitted:
    X_row = build_feature_row()
    X_row_scaled = scaler.transform(X_row)                     # same scaling as training
    prob = float(log_model.predict_proba(X_row_scaled)[0, 1])  # probability of churn

    if prob < LOW_CUTOFF:
        band, icon, advice = "Low", "🟢", "Healthy customer. No special action needed."
    elif prob < HIGH_CUTOFF:
        band, icon, advice = "Medium", "🟠", "Keep an eye on this customer. Consider a check-in or a usage tip."
    else:
        band, icon, advice = "High", "🔴", "High risk. Prioritise a retention call or offer."

    # --- Guardrail: warn if any input is outside what the model saw in training ---
    out_of_range = []
    if feature_ranges is not None:
        checks = {
            "Age": ("age", age),
            "Tenure (months)": ("tenure_months", tenure_months),
            "Transactions per month": ("monthly_txn", monthly_txn),
            "Average transaction amount (₦)": ("avg_txn_amount", avg_txn_amount),
            "Failed transactions (%)": ("failed_txn_rate", failed_txn_pct / 100.0),
            "Support tickets": ("support_tickets", support_tickets),
            "Days since last login": ("days_since_last_login", days_since_last_login),
        }
        for label, (col, value) in checks.items():
            if col not in feature_ranges:
                continue
            lo, hi = feature_ranges[col]["min"], feature_ranges[col]["max"]
            if value < lo or value > hi:
                scale = 100.0 if col == "failed_txn_rate" else 1.0   # show the rate as a %
                out_of_range.append(
                    f"**{label}**: you entered {value * scale:,.1f}, "
                    f"but the model only saw {lo * scale:,.1f} to {hi * scale:,.1f}"
                )

    st.divider()
    st.subheader("Result")
    if out_of_range:
        st.warning(
            "⚠️ **Treat this score as unreliable.** These inputs are outside the range the model "
            "was trained on, so it is guessing:\n\n- " + "\n- ".join(out_of_range)
        )
    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk band", f"{icon} {band}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.info(advice)

    # --- Why this score: coefficient x scaled value for each feature ---
    contributions = log_model.coef_[0] * X_row_scaled[0]
    drivers = pd.DataFrame({
        "Factor": [FRIENDLY_NAMES.get(c, c) for c in feature_columns],
        "Push": contributions,
    })
    drivers["Direction"] = np.where(drivers["Push"] > 0, "Raises risk", "Lowers risk")
    drivers = drivers.reindex(drivers["Push"].abs().sort_values(ascending=False).index).head(5)

    st.subheader("What is driving this score")
    st.dataframe(
        drivers[["Factor", "Direction", "Push"]].round({"Push": 2}),
        hide_index=True,
    )
    st.caption("Push = model weight x how far this customer is from the average. "
               "Bigger absolute numbers matter more.")

    with st.expander("Show the exact values sent to the model"):
        st.dataframe(X_row.T.rename(columns={0: "value"}))
else:
    st.info("Fill in the customer details and press **Predict churn risk**.")