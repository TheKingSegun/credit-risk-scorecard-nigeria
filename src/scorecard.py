"""
Credit Risk Scorecard — WoE/IV feature engineering + logistic regression
Nigerian lending context
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

SCORE_OFFSET = 600
SCORE_FACTOR = 40

FEATURES = [
    "bureau_score", "debt_to_income", "employment_months",
    "num_existing_loans", "monthly_income_band", "account_age_months",
    "bvn_verified", "mobile_money_active", "has_informal_income",
]

def compute_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    """Compute Information Value for a feature."""
    temp = df[[feature, target]].copy()
    if df[feature].nunique() > bins:
        temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")
    else:
        temp["bin"] = temp[feature]

    grouped = temp.groupby("bin")[target].agg(["sum", "count"])
    grouped.columns = ["bads", "total"]
    grouped["goods"] = grouped["total"] - grouped["bads"]
    total_bads = grouped["bads"].sum()
    total_goods = grouped["goods"].sum()
    grouped["dist_bad"] = grouped["bads"] / (total_bads + 1e-9)
    grouped["dist_good"] = grouped["goods"] / (total_goods + 1e-9)
    grouped["woe"] = np.log((grouped["dist_good"] + 1e-9) / (grouped["dist_bad"] + 1e-9))
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    return grouped["iv"].sum()

def rank_features_by_iv(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """Rank all features by Information Value."""
    results = []
    for f in features:
        iv = compute_iv(df, f, target)
        power = "Strong" if iv > 0.3 else "Medium" if iv > 0.1 else "Weak"
        results.append({"feature": f, "iv": round(iv, 4), "predictive_power": power})
    return pd.DataFrame(results).sort_values("iv", ascending=False)

def scale_to_scorecard(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds to 300-850 credit score scale."""
    return np.clip(SCORE_OFFSET + SCORE_FACTOR * (log_odds / np.log(2)), 300, 850).astype(int)

def generate_sample_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic Nigerian lending applicant data."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "bureau_score": np.random.normal(550, 100, n).clip(300, 850),
        "debt_to_income": np.random.beta(2, 5, n),
        "employment_months": np.random.exponential(36, n).clip(0, 360),
        "num_existing_loans": np.random.poisson(1.5, n).clip(0, 8),
        "monthly_income_band": np.random.randint(1, 6, n),
        "account_age_months": np.random.exponential(24, n).clip(1, 240),
        "bvn_verified": np.random.binomial(1, 0.82, n),
        "mobile_money_active": np.random.binomial(1, 0.65, n),
        "has_informal_income": np.random.binomial(1, 0.38, n),
    })
    log_odds = (
        -2.5
        + 0.008 * (df["bureau_score"] - 550)
        - 3.0 * df["debt_to_income"]
        + 0.008 * df["employment_months"]
        - 0.2 * df["num_existing_loans"]
        + 0.15 * df["monthly_income_band"]
        + 0.004 * df["account_age_months"]
        + 0.3 * df["bvn_verified"]
        + np.random.normal(0, 0.5, n)
    )
    df["default"] = (1 / (1 + np.exp(log_odds)) > np.random.uniform(0, 1, n)).astype(int)
    return df

class NigerianCreditScorecard:
    def __init__(self, cutoff_score: int = 560):
        self.model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.cutoff_score = cutoff_score
        self.selected_features = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target: str = "default"):
        iv_df = rank_features_by_iv(df, FEATURES, target)
        self.selected_features = iv_df[iv_df["iv"] >= 0.1]["feature"].tolist()
        X = self.scaler.fit_transform(df[self.selected_features])
        y = df[target]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc").mean()
        print(f"CV ROC-AUC: {cv_auc:.3f}")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def score(self, applicant: Dict) -> Dict:
        X = pd.DataFrame([applicant])[self.selected_features]
        X_scaled = self.scaler.transform(X)
        log_odds = self.model.decision_function(X_scaled)[0]
        credit_score = int(scale_to_scorecard(np.array([log_odds]))[0])
        default_prob = float(self.model.predict_proba(X_scaled)[0][1])
        decision = "APPROVE" if credit_score >= self.cutoff_score else "DECLINE"
        risk_band = "Low Risk" if credit_score >= 680 else "Medium Risk" if credit_score >= 580 else "High Risk"
        return {
            "credit_score": credit_score,
            "default_probability": round(default_prob, 4),
            "decision": decision,
            "risk_band": risk_band,
        }

if __name__ == "__main__":
    df = generate_sample_data()
    sc = NigerianCreditScorecard()
    sc.fit(df)
    result = sc.score({
        "bureau_score": 620, "debt_to_income": 0.35, "employment_months": 24,
        "num_existing_loans": 1, "monthly_income_band": 3, "account_age_months": 18,
        "bvn_verified": 1, "mobile_money_active": 1, "has_informal_income": 0,
    })
    print(result)
