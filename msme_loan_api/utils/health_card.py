# ============================================================
# utils/health_card.py
# Financial Health Card — 5-Dimension Scoring Logic
# ============================================================

from schemas.loan_schema import HealthCardScores


def compute_health_card(applicant: dict) -> HealthCardScores:
    """
    Compute 5-dimension Financial Health Score (0–100 each).
    Overall score = weighted average on 0–1000 scale.
    """

    # ── 1. Liquidity Score ───────────────────────────────────
    # Lower EMI-to-income ratio = better liquidity
    emi_ratio  = applicant.get('emi_to_income_ratio', 0.5)
    liquidity  = max(0, min(100, round((1 - emi_ratio) * 100)))

    # ── 2. Solvency Score ────────────────────────────────────
    # Lower DTI + lower loan-to-income = better solvency
    dti        = applicant.get('dti', 30)
    lti        = applicant.get('loan_to_income_ratio', 0.5)
    solvency   = max(0, min(100, round(100 - (dti * 1.5) - (lti * 20))))

    # ── 3. Growth Score ──────────────────────────────────────
    # Higher annual income = better growth potential
    income     = applicant.get('annual_income', 30000)
    growth     = max(0, min(100, round(min(income / 1000, 100))))

    # ── 4. Compliance Score ──────────────────────────────────
    # Verified docs + better grade = better compliance
    verify     = applicant.get('verification_status_encoded', 0)
    grade      = applicant.get('grade_encoded', 3)
    compliance = max(0, min(100, round((verify / 2) * 50 + (grade / 7) * 50)))

    # ── 5. Repayment Score ───────────────────────────────────
    # Higher payment ratio + recent payments = better repayment
    pay_ratio  = applicant.get('payment_ratio', 0)
    days_late  = applicant.get('days_since_last_payment', 30)
    repayment  = max(0, min(100, round(
        (pay_ratio * 60) + max(0, (1 - days_late / 365)) * 40
    )))

    # ── Overall Score (0–1000) ───────────────────────────────
    weights = {
        'liquidity'  : 0.25,
        'solvency'   : 0.25,
        'growth'     : 0.15,
        'compliance' : 0.20,
        'repayment'  : 0.15
    }
    scores = {
        'liquidity'  : liquidity,
        'solvency'   : solvency,
        'growth'     : growth,
        'compliance' : compliance,
        'repayment'  : repayment
    }
    overall = round(sum(scores[k] * weights[k] for k in scores) * 10)

    return HealthCardScores(
        liquidity  = liquidity,
        solvency   = solvency,
        growth     = growth,
        compliance = compliance,
        repayment  = repayment,
        overall    = overall
    )


def get_risk_band(probability: float) -> str:
    """Map default probability to risk band."""
    if probability < 0.35:
        return "GREEN"
    elif probability < 0.65:
        return "AMBER"
    else:
        return "RED"


def get_decision(probability: float) -> str:
    """Map default probability to loan decision."""
    if probability < 0.35:
        return "APPROVE"
    elif probability < 0.65:
        return "MANUAL REVIEW"
    else:
        return "REJECT"


def compute_loan_offer(
    requested_amount : float,
    requested_rate   : float,
    requested_term   : int,
    probability      : float,
    health_score     : int
) -> dict:
    """
    Compute recommended loan offer based on risk.
    Higher risk → lower amount, higher rate.
    """
    if probability < 0.35:
        # Low risk — full offer
        amount = requested_amount
        rate   = requested_rate
        term   = requested_term

    elif probability < 0.65:
        # Medium risk — reduce amount, increase rate slightly
        amount = round(requested_amount * 0.75, 2)
        rate   = round(requested_rate   * 1.15, 2)
        term   = requested_term

    else:
        # High risk — no offer
        return {}

    return {
        "recommended_amount" : amount,
        "recommended_rate"   : rate,
        "recommended_term"   : term
    }
