"""
Credit Risk Scorecard — Interactive Streamlit Demo
Nigerian Retail Lending | Vantage Analytics
Scorecard logic is self-contained — no import path issues on Streamlit Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

st.set_page_config(page_title="Credit Risk Scorecard | Nigeria", page_icon="📋", layout="wide")

FEATURES = [
    "bureau_score","debt_to_income","employment_months",
    "num_existing_loans","monthly_income_band","account_age_months",
    "bvn_verified","mobile_money_active","has_informal_income",
]

def compute_iv(df, feature, target, bins=10):
    temp = df[[feature, target]].copy()
    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop") if df[feature].nunique() > bins else temp[feature]
    g = temp.groupby("bin", observed=True)[target].agg(["sum","count"])
    g.columns = ["bads","total"]
    g["goods"] = g["total"] - g["bads"]
    tb, tg = g["bads"].sum(), g["goods"].sum()
    g["db"] = g["bads"]  / (tb + 1e-9)
    g["dg"] = g["goods"] / (tg + 1e-9)
    g["woe"] = np.log((g["dg"] + 1e-9) / (g["db"] + 1e-9))
    g["iv"]  = (g["dg"] - g["db"]) * g["woe"]
    return g["iv"].sum()

def rank_iv(df, features, target):
    rows = []
    for f in features:
        iv = compute_iv(df, f, target)
        rows.append({"feature":f,"iv":round(iv,4),"predictive_power":"Strong" if iv>0.3 else "Medium" if iv>0.1 else "Weak"})
    return pd.DataFrame(rows).sort_values("iv", ascending=False)

def to_score(log_odds):
    return np.clip(600 + 40*(log_odds/np.log(2)), 300, 850).astype(int)

def gen_data(n=5000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "bureau_score":        np.random.normal(550,100,n).clip(300,850),
        "debt_to_income":      np.random.beta(2,5,n),
        "employment_months":   np.random.exponential(36,n).clip(0,360),
        "num_existing_loans":  np.random.poisson(1.5,n).clip(0,8),
        "monthly_income_band": np.random.randint(1,6,n).astype(float),
        "account_age_months":  np.random.exponential(24,n).clip(1,240),
        "bvn_verified":        np.random.binomial(1,0.82,n).astype(float),
        "mobile_money_active": np.random.binomial(1,0.65,n).astype(float),
        "has_informal_income": np.random.binomial(1,0.38,n).astype(float),
    })
    lo = (-2.5 + 0.008*(df["bureau_score"]-550) - 3.0*df["debt_to_income"]
          + 0.008*df["employment_months"] - 0.2*df["num_existing_loans"]
          + 0.15*df["monthly_income_band"] + 0.004*df["account_age_months"]
          + 0.3*df["bvn_verified"] + np.random.normal(0,0.5,n))
    df["default"] = (1/(1+np.exp(lo)) > np.random.uniform(0,1,n)).astype(int)
    return df

@st.cache_resource(show_spinner="Training scorecard on 5,000 applicants…")
def load_model():
    df   = gen_data()
    ivdf = rank_iv(df, FEATURES, "default")
    sel  = ivdf[ivdf["iv"] >= 0.1]["feature"].tolist()
    sc   = StandardScaler()
    X    = sc.fit_transform(df[sel])
    y    = df["default"]
    m    = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    auc  = cross_val_score(m, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="roc_auc").mean()
    m.fit(X, y)
    return m, sc, sel, ivdf, df, round(auc, 3)

model, scaler, sel_feat, iv_df, train_df, cv_auc = load_model()

def score(app):
    X   = scaler.transform(pd.DataFrame([app])[sel_feat])
    lo  = model.decision_function(X)[0]
    cs  = int(to_score(np.array([lo]))[0])
    dp  = float(model.predict_proba(X)[0][1])
    dec = "APPROVE" if cs >= 560 else "DECLINE"
    rb  = "Low Risk" if cs >= 680 else "Medium Risk" if cs >= 580 else "High Risk"
    return {"credit_score":cs,"default_probability":round(dp,4),"decision":dec,"risk_band":rb}

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Nigerian Credit Risk Scorecard")
st.markdown(f"WoE/IV logistic regression scorecard · **CV ROC-AUC: {cv_auc}** · Gini ≈ {round(2*cv_auc-1,2)} · Lagos, Nigeria")

tab1, tab2, tab3 = st.tabs(["Score Applicant","Model Performance","Feature Analysis"])

with tab1:
    st.markdown("### Applicant details")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Financial**")
        bs  = st.slider("Bureau score", 300, 850, 620)
        dti = st.slider("Debt-to-income", 0.0, 1.0, 0.35, step=0.01)
        nl  = st.slider("Existing loans", 0, 8, 1)
    with c2:
        st.markdown("**Employment & income**")
        em  = st.slider("Employment (months)", 0, 240, 24)
        ib  = st.selectbox("Monthly income", [1,2,3,4,5], index=2,
                format_func=lambda x:{1:"< ₦50k",2:"₦50k–150k",3:"₦150k–300k",4:"₦300–500k",5:"> ₦500k"}[x])
        aa  = st.slider("Account age (months)", 1, 240, 18)
    with c3:
        st.markdown("**Verification**")
        bvn = st.radio("BVN verified?",   [1,0], format_func=lambda x:"Yes" if x else "No", horizontal=True)
        mm  = st.radio("Mobile money?",   [1,0], format_func=lambda x:"Yes" if x else "No", horizontal=True)
        inf = st.radio("Informal income?",[0,1], format_func=lambda x:"Yes" if x else "No", horizontal=True)

    st.divider()
    if st.button("Run Credit Assessment", type="primary"):
        app = {"bureau_score":bs,"debt_to_income":dti,"employment_months":em,
               "num_existing_loans":nl,"monthly_income_band":ib,"account_age_months":aa,
               "bvn_verified":bvn,"mobile_money_active":mm,"has_informal_income":inf}
        res = score(app); cs = res["credit_score"]
        cl, cr = st.columns([2,1])
        with cl:
            color = "#1D9E75" if cs >= 560 else "#D85A30"
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=cs, delta={"reference":560},
                title={"text":"Credit Score"},
                gauge={"axis":{"range":[300,850]},"bar":{"color":color,"thickness":0.3},
                       "steps":[{"range":[300,480],"color":"#FCEBEB"},{"range":[480,560],"color":"#FAEEDA"},
                                 {"range":[560,660],"color":"#E1F5EE"},{"range":[660,850],"color":"#9FE1CB"}],
                       "threshold":{"line":{"color":"#333","width":3},"value":560}}))
            fig.update_layout(height=300, margin=dict(t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Cutoff: 560 | 300–480 High Risk · 480–560 Medium · 560–660 Good · 660+ Low Risk")
        with cr:
            if res["decision"]=="APPROVE": st.success("### ✓ APPROVE")
            else: st.error("### ✗ DECLINE")
            st.metric("Credit Score", cs)
            st.metric("Default Probability", f"{res['default_probability']:.1%}")
            st.metric("Risk Band", res["risk_band"])
            reasons = []
            if dti > 0.5:  reasons.append("High debt-to-income")
            if em  < 12:   reasons.append("Short employment tenure")
            if nl  >= 3:   reasons.append("Multiple existing loans")
            if bs  < 500:  reasons.append("Low bureau score")
            if not bvn:    reasons.append("BVN not verified")
            if not reasons: reasons = ["No major adverse factors"]
            st.markdown("**Risk factors:**")
            for r in reasons[:3]: st.markdown(f"- {r}")

with tab2:
    st.markdown("### Validation metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CV ROC-AUC", str(cv_auc))
    c2.metric("Gini", str(round(2*cv_auc-1,2)))
    c3.metric("Features selected", len(sel_feat))
    c4.metric("Training sample", "5,000")

    samp = train_df.sample(min(500,len(train_df)), random_state=42)
    rows = []
    for _, row in samp.iterrows():
        try:
            r = score({f:row[f] for f in sel_feat})
            rows.append({**r,"actual_default":int(row["default"])})
        except: pass
    sdf = pd.DataFrame(rows)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(sdf, x="credit_score",
            color=sdf["actual_default"].map({0:"Good",1:"Bad"}),
            barmode="overlay", nbins=25, opacity=0.75,
            color_discrete_map={"Good":"#1D9E75","Bad":"#D85A30"},
            title="Score distribution: good vs. bad accounts")
        fig.update_layout(height=300, legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bins_  = pd.cut(sdf["credit_score"], bins=6)
        rates_ = sdf.groupby(bins_, observed=True)["actual_default"].mean().reset_index()
        rates_.columns = ["band","default_rate"]
        rates_["band"] = rates_["band"].astype(str)
        fig = px.bar(rates_, x="band", y="default_rate", color="default_rate",
                     color_continuous_scale="RdYlGn_r",
                     text=rates_["default_rate"].apply(lambda x:f"{x:.1%}"),
                     title="Default rate by score band")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Feature Information Values")
    st.markdown("**IV > 0.3** = strong · **0.1–0.3** = medium · **< 0.1** = excluded")
    fig = px.bar(iv_df, x="iv", y="feature", orientation="h", color="iv",
                 color_continuous_scale="Greens",
                 text=iv_df["iv"].apply(lambda x:f"{x:.3f}"),
                 title="Predictive power by feature (Information Value)")
    fig.add_vline(x=0.1, line_dash="dash", line_color="#D85A30", annotation_text="Selection threshold")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(iv_df.rename(columns={"feature":"Feature","iv":"IV","predictive_power":"Power"}),
                 hide_index=True, use_container_width=True)

st.divider()
st.caption("David · Vantage Analytics · github.com/TheKingSegun/credit-risk-scorecard-nigeria")
