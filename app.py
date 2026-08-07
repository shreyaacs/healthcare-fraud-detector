import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="Healthcare Fraud Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        margin-top: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Data Loading & Preprocessing Pipeline
@st.cache_data
def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_path(filename):
        root_path = os.path.join(base_dir, filename)
        if os.path.exists(root_path):
            return root_path

        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)

        raise FileNotFoundError(
            f"Could not locate '{filename}' in project folder."
        )

    train_target = pd.read_csv(get_path("Train-1542865627584.csv"))
    inpatient_train = pd.read_csv(
        get_path("Train_Inpatientdata-1542865627584.csv")
    )
    outpatient_train = pd.read_csv(
        get_path("Train_Outpatientdata-1542865627584.csv")
    )
    bene_train = pd.read_csv(
        get_path("Train_Beneficiarydata-1542865627584.csv")
    )

    unseen_target = pd.read_csv(get_path("Unseen-1542969243754.csv"))
    inpatient_unseen = pd.read_csv(
        get_path("Unseen_Inpatientdata-1542969243754.csv")
    )
    outpatient_unseen = pd.read_csv(
        get_path("Unseen_Outpatientdata-1542969243754.csv")
    )
    bene_unseen = pd.read_csv(
        get_path("Unseen_Beneficiarydata-1542969243754.csv")
    )

    def aggregate_data(inpatient_df, outpatient_df, bene_df):
        inpatient_df = inpatient_df.copy()
        outpatient_df = outpatient_df.copy()
        inpatient_df["is_inpatient"] = 1
        outpatient_df["is_inpatient"] = 0

        claims_df = pd.concat([inpatient_df, outpatient_df], ignore_index=True)

        chronic_cols = [
            c for c in bene_df.columns if c.startswith("ChronicCond_")
        ]
        bene_df = bene_df.copy()
        bene_df["num_chronic_conditions"] = (
            bene_df[chronic_cols].replace(2, 0).sum(axis=1)
        )

        claims_bene_df = pd.merge(claims_df, bene_df, on="BeneID", how="left")

        claims_bene_df["ClaimStartDt"] = pd.to_datetime(
            claims_bene_df["ClaimStartDt"]
        )
        claims_bene_df["DOB"] = pd.to_datetime(claims_bene_df["DOB"])
        claims_bene_df["AdmissionDt"] = pd.to_datetime(
            claims_bene_df["AdmissionDt"]
        )
        claims_bene_df["DischargeDt"] = pd.to_datetime(
            claims_bene_df["DischargeDt"]
        )

        claims_bene_df["patient_age"] = (
            claims_bene_df["ClaimStartDt"] - claims_bene_df["DOB"]
        ).dt.days / 365.25
        claims_bene_df["length_of_stay"] = (
            claims_bene_df["DischargeDt"] - claims_bene_df["AdmissionDt"]
        ).dt.days + 1

        claims_bene_df["attending_is_operating"] = (
            (claims_bene_df["AttendingPhysician"].notna())
            & (
                claims_bene_df["AttendingPhysician"]
                == claims_bene_df["OperatingPhysician"]
            )
        ).astype(int)

        grouped = claims_bene_df.groupby("Provider")

        provider_agg = grouped.agg(
            total_claims=("ClaimID", "count"),
            total_reimbursement=("InscClaimAmtReimbursed", "sum"),
            mean_reimbursement=("InscClaimAmtReimbursed", "mean"),
            max_reimbursement=("InscClaimAmtReimbursed", "max"),
            total_deductible=("DeductibleAmtPaid", "sum"),
            inpatient_claims=("is_inpatient", "sum"),
            unique_patients=("BeneID", "nunique"),
            unique_attending_physicians=("AttendingPhysician", "nunique"),
            unique_operating_physicians=("OperatingPhysician", "nunique"),
            unique_other_physicians=("OtherPhysician", "nunique"),
            attending_is_operating_count=("attending_is_operating", "sum"),
            avg_patient_age=("patient_age", "mean"),
            avg_chronic_conditions=("num_chronic_conditions", "mean"),
            avg_length_of_stay=("length_of_stay", "mean"),
        ).reset_index()

        provider_agg["outpatient_claims"] = (
            provider_agg["total_claims"] - provider_agg["inpatient_claims"]
        )
        provider_agg["inpatient_ratio"] = (
            provider_agg["inpatient_claims"] / provider_agg["total_claims"]
        )

        phys_cols = [
            "AttendingPhysician",
            "OperatingPhysician",
            "OtherPhysician",
        ]
        unique_total_physicians = (
            claims_bene_df[["Provider"] + phys_cols]
            .melt(id_vars=["Provider"], value_vars=phys_cols)
            .dropna()
            .groupby("Provider")["value"]
            .nunique()
            .reset_index(name="unique_total_physicians")
        )

        provider_agg = pd.merge(
            provider_agg, unique_total_physicians, on="Provider", how="left"
        )
        provider_agg["unique_total_physicians"] = (
            provider_agg["unique_total_physicians"].fillna(0).astype(int)
        )

        return provider_agg

    train_agg = aggregate_data(inpatient_train, outpatient_train, bene_train)
    train_target["PotentialFraud"] = train_target["PotentialFraud"].map(
        {"Yes": 1, "No": 0}
    )
    train_full = pd.merge(train_agg, train_target, on="Provider", how="left")

    unseen_agg = aggregate_data(inpatient_unseen, outpatient_unseen, bene_unseen)
    unseen_full = pd.merge(unseen_target, unseen_agg, on="Provider", how="left")

    to_drop = [
        "total_deductible",
        "inpatient_claims",
        "unique_patients",
        "unique_operating_physicians",
        "unique_other_physicians",
        "attending_is_operating_count",
        "outpatient_claims",
        "unique_total_physicians",
    ]

    train_reduced = train_full.drop(columns=to_drop)
    unseen_reduced = unseen_full.drop(columns=to_drop, errors="ignore")

    X_train = train_reduced.drop(
        columns=["Provider", "PotentialFraud"]
    ).fillna(0)
    y_train = train_reduced["PotentialFraud"]

    feature_cols = X_train.columns.tolist()
    X_unseen = unseen_reduced[feature_cols].fillna(0)

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    val_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    val_model.fit(X_tr, y_tr)
    val_probs = val_model.predict_proba(X_val)[:, 1]

    best_thresh = 0.32
    unseen_probs = model.predict_proba(X_unseen)[:, 1]
    unseen_preds = (unseen_probs >= best_thresh).astype(int)

    unseen_full["Probability"] = np.round(unseen_probs, 4)
    unseen_full["Predicted Class"] = unseen_preds

    submission_df = pd.DataFrame(
        {
            "Provider": unseen_full["Provider"],
            "Probability": unseen_full["Probability"],
            "Predicted Class": unseen_full["Predicted Class"],
        }
    )

    return (
        train_full,
        unseen_full,
        submission_df,
        model,
        feature_cols,
        y_val,
        val_probs,
    )


(
    train_full,
    unseen_full,
    submission_df,
    model,
    feature_cols,
    y_val,
    val_probs,
) = load_and_preprocess_data()



st.title("Healthcare Provider Fraud Dashboard")
st.markdown(" Track, analyze, and check healthcare providers for potential insurance fraud.")

# Simplified Tab Names
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview & Charts",
        " Model Performance",
        " Check Provider Risk",
        " Download Results",
    ]
)


# TAB 1: OVERVIEW & CHARTS


with tab1:
    st.header("Executive Summary")

    total_providers = len(train_full)
    fraud_count = train_full["PotentialFraud"].sum()
    fraud_rate = (fraud_count / total_providers) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Providers Analyzed", f"{total_providers:,}")
    col2.metric("Fraudulent Providers Detected", f"{fraud_count:,}")
    col3.metric("Overall Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Unseen Test Providers", f"{len(unseen_full):,}")

    st.markdown("---")
    st.subheader("Interactive Exploratory Data Analysis")

    train_full["Fraud Label"] = train_full["PotentialFraud"].map(
        {1: "Fraudulent", 0: "Normal"}
    )

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("##### Average Money Claimed per Provider ($)")
       
        fig_box = px.box(
            train_full,
            x="Fraud Label",
            y="mean_reimbursement",
            color="Fraud Label",
            color_discrete_map={"Fraudulent": "#e74c3c", "Normal": "#2ecc71"},
            points="outliers",
            title="Mean Reimbursement ($) by Provider Type",
            labels={
                "mean_reimbursement": "Mean Reimbursement ($)",
                "Fraud Label": "Provider Type",
            },
        )
       
        fig_box.update_yaxes(range=[0, 15000])
        fig_box.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)

    with eda_col2:
        st.markdown("##### Total Claims Submitted per Provider")
        
        fig_violin = px.violin(
            train_full,
            x="Fraud Label",
            y="total_claims",
            color="Fraud Label",
            color_discrete_map={"Fraudulent": "#e74c3c", "Normal": "#2ecc71"},
            box=True,
            points=False,
            log_y=True,  # Log scale transforms the compressed shape into a clear visual comparison
            title="Total Claims Processed (Log Scale)",
            labels={
                "total_claims": "Total Claims (Log Scale)",
                "Fraud Label": "Provider Type",
            },
        )
        fig_violin.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_violin, use_container_width=True)

   
    st.markdown("---")
    st.markdown("##### Key Behavioral Metric Comparisons (Medians)")
    
    avg_df = (
        train_full.groupby("Fraud Label")[
            ["mean_reimbursement", "total_claims", "unique_patients"]
        ]
        .median()
        .reset_index()
    )

    bar_col1, bar_col2 = st.columns(2)

    with bar_col1:
        fig_bar_reimb = px.bar(
            avg_df,
            x="Fraud Label",
            y="mean_reimbursement",
            color="Fraud Label",
            color_discrete_map={"Fraudulent": "#e74c3c", "Normal": "#2ecc71"},
            text_auto=".2f",
            title="Median Reimbursement Amount ($)",
            labels={
                "mean_reimbursement": "Median Amount ($)",
                "Fraud Label": "Provider Type",
            },
        )
        fig_bar_reimb.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_bar_reimb, use_container_width=True)

    with bar_col2:
        fig_bar_claims = px.bar(
            avg_df,
            x="Fraud Label",
            y="total_claims",
            color="Fraud Label",
            color_discrete_map={"Fraudulent": "#e74c3c", "Normal": "#2ecc71"},
            text_auto=True,
            title="Median Total Claims Processed",
            labels={
                "total_claims": "Median Claims Count",
                "Fraud Label": "Provider Type",
            },
        )
        fig_bar_claims.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_bar_claims, use_container_width=True)


# TAB 2: MODEL PERFORMANCE
with tab2:
    st.header("How Well the AI Model Performs")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.subheader("Most Important Factors for Spotting Fraud")
        importances = pd.DataFrame(
            {"Feature": feature_cols, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Key Indicators Used by Model",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Importance Score", "Feature": "Claim Feature"},
        )
        fig_imp.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with perf_col2:
        st.subheader("Model Accuracy Curve (ROC Curve)")
        fpr, tpr, _ = roc_curve(y_val, val_probs)
        auc_score = roc_auc_score(y_val, val_probs)
        f1_val = f1_score(y_val, (val_probs >= 0.32).astype(int))

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Our Model (Score = {auc_score:.2f})",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title=f"Accuracy Score: {auc_score:.2f} out of 1.0",
            xaxis_title="False Alarm Rate",
            yaxis_title="True Fraud Detection Rate",
            template="plotly_white",
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# TAB 3: CHECK PROVIDER RISK
with tab3:
    st.header("Look Up a Provider")

    provider_list = unseen_full["Provider"].unique().tolist()
    selected_provider = st.selectbox(
        "Choose or type a Provider ID:", provider_list
    )

    if selected_provider:
        p_data = unseen_full[
            unseen_full["Provider"] == selected_provider
        ].iloc[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            risk_prob = p_data["Probability"]
            pred_class = p_data["Predicted Class"]

            st.metric("Fraud Risk Score", f"{risk_prob * 100:.1f}%")
            if pred_class == 1:
                st.error(" HIGH RISK: Likely Fraudulent Provider!")
            else:
                st.success(" LOW RISK: Normal Provider")

        with res_col2:
            st.subheader("Provider Details")
            metrics_json = {
                "Provider ID": p_data["Provider"],
                "Total Claims": int(p_data["total_claims"]),
                "Total Money Claimed": f"${p_data['total_reimbursement']:,.2f}",
                "Average Claim Amount": f"${p_data['mean_reimbursement']:,.2f}",
                "Inpatient Care Ratio": f"{p_data['inpatient_ratio'] * 100:.1f}%",
                "Number of Patients": int(p_data["unique_patients"]),
                "Number of Main Doctors": int(
                    p_data["unique_attending_physicians"]
                ),
                "Times Main Doctor Was Also Surgeon": int(
                    p_data["attending_is_operating_count"]
                ),
                "Average Patient Age": f"{p_data['avg_patient_age']:.1f} years",
                "Average Health Conditions per Patient": (
                    f"{p_data['avg_chronic_conditions']:.1f}"
                ),
            }
            st.json(metrics_json)


# TAB 4: DOWNLOAD RESULTS
with tab4:
    st.header("Download Predictions")

    st.markdown("##### Preview Prediction Results for New Providers")
    st.dataframe(submission_df, use_container_width=True, height=400)

    csv_data = submission_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results (CSV)",
        data=csv_data,
        file_name="Fraud_Predictions_Report.csv",
        mime="text/csv",
        help="Click to download the prediction report as a CSV file.",
    )