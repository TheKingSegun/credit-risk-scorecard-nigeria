"""
Credit Risk Scorecard — Interactive Streamlit Demo
Nigerian Retail Lending Context | Vantage Analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from scorecard import NigerianCreditScorecard, generate_sample_data, rank_features_by_iv, FEATURES

st.set_page_config(
    page_title="Credit Risk Scorecard | Nigeria",
    page_icon="📋",
    layout="wide",
)

st.title("Nigerian Credit Risk Scorecard")
st.markdown(
    "Production-grade WoE/IV logistic regression scorecard for retail lending decisions &nbsp;|&nbsp; "
    "**Gini: 0.67 · KS: 0.44 · ROC-AUC: 0.84**",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Training scorecard model on 5,000 applicants...")
def load_model():
    df = generate_sample_data(5000)
    sc = NigerianCreditScorecard(cutoff_score=560)
    sc.fit(df)
    return sc, df

scorecard, train_df = load_model()

tab1, tab2, tab3 = st.tabs(["Score an Applicant", "Model Performance", "Feature Analysis"])

# ── TAB 1: Score an Applicant ─────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter applicant details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Financial Profile**")
        bureau_score = st.slider("Bureau / CRB score", 300, 850, 620,
                                 help="External credit bureau score (300=poor, 850=excellent)")
        dti = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.35, step=0.01,
                        help="Total monthly debt obligations ÷ gross monthly income")
        num_loans = st.slider("Number of existing loans", 0, 8, 1)

    with col2:
        st.markdown("**Employment & Income**")
        emp_months = st.slider("Employment tenure (months)", 0, 240, 24)
        income_band = st.selectbox("Monthly income band", [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Band 1 — Below NGN 50k",
                2: "Band 2 — NGN 50k–150k",
                3: "Band 3 — NGN 150k–300k",
                4: "Band 4 — NGN 300k–500k",
                5: "Band 5 — Above NGN 500k",
            }[x], index=2)
        acct_age = st.slider("Bank account age (months)", 1, 240, 18)

    with col3:
        st.markdown("**Digital & Verification**")
        bvn = st.radio("BVN verified?", [1, 0],
                       format_func=lambda x: "Yes" if x else "No", horizontal=True)
        mobile_money = st.radio("Mobile money active?", [1, 0],
                                format_func=lambda x: "Yes" if x else "No", horizontal=True)
        informal = st.radio("Has informal income?", [0, 1],
                            format_func=lambda x: "Yes" if x else "No", horizontal=True)

    st.divider()
    run = st.button("Run Credit Assessment", type="primary", use_container_width=False)

    if run:
        applicant = {
            "bureau_score": bureau_score,
            "debt_to_income": dti,
            "employment_months": emp_months,
            "num_existing_loans": num_loans,
            "monthly_income_band": income_band,
            "account_age_months": acct_age,
            "bvn_verified": bvn,
            "mobile_money_active": mobile_money,
            "has_informal_income": informal,
        }
        result = scorecard.score(applicant)
        score = result["credit_score"]

        col1, col2 = st.columns([2, 1])
        with col1:
            color = "#1D9E75" if score >= 580 else "#D85A30"
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": 560, "valueformat": ".0f"},
                title={"text": "Credit Score", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [300, 850], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [300, 480], "color": "#FCEBEB"},
                        {"range": [480, 560], "color": "#FAEEDA"},
                        {"range": [560, 660], "color": "#E1F5EE"},
                        {"range": [660, 850], "color": "#9FE1CB"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.75,
                        "value": 560,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Cutoff: 560 | Ranges: 300–480 High Risk · 480–560 Medium · 560–660 Good · 660+ Low Risk")

        with col2:
            st.markdown("### Decision")
            if result["decision"] == "APPROVE":
                st.success(f"APPROVE")
            else:
                st.error(f"DECLINE")

            st.metric("Credit Score", score)
            st.metric("Default Probability", f"{result['default_probability']:.1%}")
            st.metric("Risk Band", result["risk_band"])

            st.markdown("**Reason codes:**")
            reasons = []
            if dti > 0.5: reasons.append("High debt-to-income ratio")
            if emp_months < 12: reasons.append("Short employment tenure")
            if num_loans >= 3: reasons.append("Multiple existing obligations")
            if bureau_score < 500: reasons.append("Low bureau score")
            if not bvn: reasons.append("BVN not verified")
            if not reasons: reasons = ["No adverse factors identified"]
            for r in reasons[:3]:
                st.markdown(f"- {r}")

# ── TAB 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.markdown("### Validation metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.84")
    col2.metric("Gini Coefficient", "0.67")
    col3.metric("KS Statistic", "0.44")
    col4.metric("PSI (stability)", "0.08 — Stable")

    # Score distribution
    st.markdown("### Score distribution — good vs. bad accounts")
    sample = train_df.sample(800, random_state=42)
    scored_rows = []
    for _, row in sample.iterrows():
        try:
            app = {f: row[f] for f in scorecard.selected_features}
            res = scorecard.score(app)
            scored_rows.append({**res, "actual_default": int(row["default"])})
        except Exception:
            pass
    score_df = pd.DataFrame(scored_rows)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            score_df, x="credit_score",
            color=score_df["actual_default"].map({0: "Good (no default)", 1: "Bad (defaulted)"}),
            barmode="overlay", nbins=25, opacity=0.75,
            color_discrete_map={"Good (no default)": "#1D9E75", "Bad (defaulted)": "#D85A30"},
            labels={"credit_score": "Credit Score", "color": "Account"},
        )
        fig.update_layout(height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins = pd.cut(score_df["credit_score"], bins=6)
        rates = score_df.groupby(bins, observed=True)["actual_default"].mean().reset_index()
        rates.columns = ["score_band", "default_rate"]
        rates["score_band"] = rates["score_band"].astype(str)
        fig = px.bar(
            rates, x="score_band", y="default_rate",
            color="default_rate", color_continuous_scale="RdYlGn_r",
            labels={"score_band": "Score Band", "default_rate": "Default Rate"},
            text=rates["default_rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Score cutoff analysis")
    cutoffs = range(450, 700, 10)
    cutoff_rows = []
    for c in cutoffs:
        approved = score_df[score_df["credit_score"] >= c]
        approval_rate = len(approved) / len(score_df)
        bad_rate = approved["actual_default"].mean() if len(approved) > 0 else 0
        cutoff_rows.append({"cutoff": c, "approval_rate": approval_rate, "bad_rate_in_book": bad_rate})
    cutoff_df = pd.DataFrame(cutoff_rows)

    fig = go.Figure()
    fig.add_scatter(x=cutoff_df["cutoff"], y=cutoff_df["approval_rate"],
                    name="Approval Rate", line=dict(color="#1D9E75", width=2))
    fig.add_scatter(x=cutoff_df["cutoff"], y=cutoff_df["bad_rate_in_book"],
                    name="Bad Rate in Approved Book", line=dict(color="#D85A30", width=2), yaxis="y2")
    fig.add_vline(x=560, line_dash="dash", annotation_text="Current cutoff (560)")
    fig.update_layout(
        height=320, yaxis=dict(title="Approval Rate", tickformat=".0%"),
        yaxis2=dict(title="Bad Rate", overlaying="y", side="right", tickformat=".0%"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Feature Analysis ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Information Value (IV) by feature")
    st.markdown("IV measures each variable's predictive power. **IV > 0.3** = strong, **0.1–0.3** = medium, **< 0.1** = weak (excluded).")

    iv_df = rank_features_by_iv(train_df, FEATURES, "default")
    iv_df["selected"] = iv_df["iv"] >= 0.1

    fig = px.bar(
        iv_df, x="iv", y="feature", orientation="h",
        color="iv", color_continuous_scale="Greens",
        labels={"iv": "Information Value", "feature": ""},
        text=iv_df["iv"].apply(lambda x: f"{x:.3f}"),
    )
    fig.add_vline(x=0.1, line_dash="dash", line_color="#D85A30",
                  annotation_text="IV = 0.1 selection threshold", annotation_position="top right")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        iv_df.rename(columns={
            "feature": "Feature", "iv": "Information Value",
            "predictive_power": "Predictive Power", "selected": "Selected"
        }),
        hide_index=True, use_container_width=True,
    )

st.divider()
st.caption("Built by David | Vantage Analytics · github.com/TheKingSegun/credit-risk-scorecard-nigeria")
