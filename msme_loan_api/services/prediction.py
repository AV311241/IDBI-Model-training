# ============================================================
# services/prediction.py
# Core Prediction Service — Feature Engineering + Inference
# ============================================================

import pandas  as pd
import numpy   as np
import shap
import joblib

from schemas.loan_schema import LoanApplicationRequest
from utils.health_card   import compute_health_card, get_risk_band, get_decision


# ── Encoding Maps ────────────────────────────────```python
GRADE_MAP = {
    'a': 7, 'b': 6, 'c': 5,
    'd': 4, 'e': 3, 'f': 2, 'g': 1
}

SUB_GRADE_LIST = [f"{g}{n}" for g in 'abcdefg' for n in range(1, 6)]
SUB_GRADE_MAP  = {sg: (len(SUB_GRADE_LIST) - i) for i, sg in enumerate(SUB_GRADE_LIST)}

HOME_MAP = {
    'own': 3, 'mortgage': 2,
    'rent': 1, 'other': 0, 'none': 0, 'any': 0
}

VERIFY_MAP = {
    'verified': 2,
    'source verified': 1,
    'not verified': 0
}

APP_TYPE_MAP = {
    'individual': 0,
    'joint app' : 1
}

TERM_MAP = {36: 0, 60: 1}


# ── State Frequency Map (from training data) ─────────────────
# Replace these values with actual frequencies from your dataset
STATE_FREQ_MAP = {
    'CA': 0.1320, 'NY': 0.0890, 'TX': 0.0760, 'FL': 0.0710,
    'IL': 0.0430, 'NJ': 0.0380, 'GA': 0.0310, 'PA': 0.0290,
    'OH': 0.0270, 'VA': 0.0250, 'NC': 0.0240, 'MI': 0.0220,
    'WA': 0.0210, 'MA': 0.0200, 'AZ': 0.0190, 'MD': 0.0180,
    'CO': 0.0170, 'MN': 0.0160, 'IN': 0.0140, 'MO': 0.0130,
}

# ── Known Purpose Categories (from training one-hot encoding) ─
KNOWN_PURPOSES = [
    'car', 'credit_card', 'debt_consolidation', 'educational',
    'home_improvement', 'house', 'major_purchase', 'medical',
    'moving', 'other', 'renewable_energy', 'small_business',
    'vacation', 'wedding'
]

# ── Known emp_title Top 20 (from training) ───────────────────
KNOWN_EMP_TITLES = [
    'teacher', 'manager', 'registered nurse', 'rn', 'supervisor',
    'driver', 'owner', 'sales', 'engineer', 'director',
    'accountant', 'analyst', 'nurse', 'consultant', 'technician',
    'officer', 'coordinator', 'specialist', 'administrator', 'other'
]


def encode_application(app: LoanApplicationRequest, feature_columns: list) -> pd.DataFrame:
    """
    Convert raw LoanApplicationRequest into a fully encoded
    DataFrame matching the training feature set exactly.
    """

    # ── Step 1: Build base encoded dict ──────────────────────
    grade_enc  = GRADE_MAP.get(app.grade.lower(), 3)
    sub_enc    = SUB_GRADE_MAP.get(app.sub_grade.lower(), 18)
    home_enc   = HOME_MAP.get(app.home_ownership.lower(), 0)
    verify_enc = VERIFY_MAP.get(app.verification_status.lower(), 0)
    app_enc    = APP_TYPE_MAP.get(app.application_type.lower(), 0)
    term_enc   = TERM_MAP.get(int(app.term), 0)
    state_enc  = STATE_FREQ_MAP.get(app.address_state.upper(), 0.01)

    # ── Step 2: Derived features ─────────────────────────────
    lti_ratio  = round(app.loan_amount / app.annual_income, 4) \
                 if app.annual_income > 0 else 0
    pay_ratio  = round(app.total_payment / app.loan_amount, 4) \
                 if app.loan_amount > 0 else 0
    emi_ratio  = round(app.installment / (app.annual_income / 12), 4) \
                 if app.annual_income > 0 else 0

    # ── Step 3: Core feature dict ─────────────────────────────
    encoded = {
        'annual_income'           : app.annual_income,
        'dti'                     : app.dti,
        'installment'             : app.installment,
        'int_rate'                : app.int_rate,
        'loan_amount'             : app.loan_amount,
        'total_acc'               : app.total_acc,
        'total_payment'           : app.total_payment,
        'emp_length'              : app.emp_length,
        'grade'                   : grade_enc,
        'sub_grade'               : sub_enc,
        'home_ownership'          : home_enc,
        'verification_status'     : verify_enc,
        'application_type'        : app_enc,
        'term'                    : term_enc,
        'address_state'           : state_enc,
        'loan_age_months'         : app.loan_age_months,
        'issue_month'             : app.issue_month,
        'issue_year'              : app.issue_year,
        'days_since_last_payment' : app.days_since_last_payment,
        'days_since_credit_pull'  : app.days_since_credit_pull,
        'days_to_next_payment'    : app.days_to_next_payment,
        'loan_to_income_ratio'    : lti_ratio,
        'payment_ratio'           : pay_ratio,
        'emi_to_income_ratio'     : emi_ratio,
    }

    # ── Step 4: One-hot encode purpose ───────────────────────
    purpose_clean = app.purpose.lower().replace(' ', '_')
    for p in KNOWN_PURPOSES[1:]:   # drop_first=True skips first category
        encoded[f'purpose_{p}'] = 1 if purpose_clean == p else 0

    # ── Step 5: One-hot encode emp_title ─────────────────────
    for t in KNOWN_EMP_TITLES[1:]:
        encoded[f'emp_title_{t}'] = 0   # default 0; set if matched
    # (emp_title not in request — default all to 0)

    # ── Step 6: Build DataFrame & align to training columns ──
    df = pd.DataFrame([encoded])

    # Add any missing training columns as 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns in exact order
    df = df[feature_columns]

    return df, {
        'grade_encoded'              : grade_enc,
        'verification_status_encoded': verify_enc,
        'loan_to_income_ratio'       : lti_ratio,
        'payment_ratio'              : pay_ratio,
        'emi_to_income_ratio'        : emi_ratio,
        'annual_income'              : app.annual_income,
        'days_since_last_payment'    : app.days_since_last_payment,
    }


def run_shap_explanation(
    explainer      : shap.TreeExplainer,
    scaled_df      : pd.DataFrame,
    feature_columns: list,
    top_n          : int = 5
) -> dict:
    """
    Compute SHAP values for a single applicant and return
    top risk-increasing and risk-decreasing features.
    """
    shap_values = explainer.shap_values(scaled_df)

    # Handle list output (binary classification)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    shap_series = pd.Series(shap_vals, index=feature_columns)

    top_risk    = shap_series.nlargest(top_n).round(4).to_dict()
    top_safety  = shap_series.nsmallest(top_n).round(4).to_dict()

    return {
        "top_risk_factors"   : top_risk,
        "top_safety_factors" : top_safety,
        "base_value"         : round(float(base_value), 4)
    }
