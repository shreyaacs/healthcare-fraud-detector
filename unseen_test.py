import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def process_and_aggregate_claims(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    bene_df: pd.DataFrame,
) -> pd.DataFrame:
    """Applies claim combination, beneficiary merging, and provider-level feature aggregation

    identically across both Train and Unseen/Test datasets.
    """
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

    phys_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
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


def generate_unseen_predictions(user_full_name: str = "John_Doe"):
    """Loads unseen test data, processes features, aligns columns, predicts probabilities,

    applies optimal thresholding, and exports to CSV.
    """
    # 1. File Paths
    train_target = pd.read_csv("Train-1542865627584.csv")
    inpatient_train = pd.read_csv("Train_Inpatientdata-1542865627584.csv")
    outpatient_train = pd.read_csv("Train_Outpatientdata-1542865627584.csv")
    bene_train = pd.read_csv("Train_Beneficiarydata-1542865627584.csv")

    unseen_target = pd.read_csv("Unseen-1542969243754.csv")
    inpatient_unseen = pd.read_csv("Unseen_Inpatientdata-1542969243754.csv")
    outpatient_unseen = pd.read_csv("Unseen_Outpatientdata-1542969243754.csv")
    bene_unseen = pd.read_csv("Unseen_Beneficiarydata-1542969243754.csv")

    # 2. Process Train and Unseen Data
    train_agg = process_and_aggregate_claims(
        inpatient_train, outpatient_train, bene_train
    )
    train_target["PotentialFraud"] = train_target["PotentialFraud"].map(
        {"Yes": 1, "No": 0}
    )
    train_full = pd.merge(train_agg, train_target, on="Provider", how="left")

    unseen_agg = process_and_aggregate_claims(
        inpatient_unseen, outpatient_unseen, bene_unseen
    )
    unseen_full = pd.merge(unseen_target, unseen_agg, on="Provider", how="left")

    # 3. Drop Collinear Features Identified in Step 2
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

    # 4. Align Unseen Features Exact with Training Matrix
    feature_cols = X_train.columns.tolist()
    X_unseen = unseen_reduced[feature_cols].fillna(0)

    # 5. Determine Optimal Probability Threshold via Validation Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    val_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    val_model.fit(X_tr, y_tr)
    val_probs = val_model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.10, 0.90, 0.02):
        score = f1_score(y_val, (val_probs >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_thresh = round(t, 2)

    print(
        f"Optimal Decision Threshold: {best_thresh} (Validation F1 Score: {best_f1:.4f})"
    )

    # 6. Fit Final Model on Complete Training Dataset
    final_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    final_model.fit(X_train, y_train)

    # 7. Predict Probabilities & Derived Classes for Unseen Providers
    unseen_probs = final_model.predict_proba(X_unseen)[:, 1]
    unseen_preds = (unseen_probs >= best_thresh).astype(int)

    # 8. Create Final Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "Provider": unseen_full["Provider"],
            "Probability": np.round(unseen_probs, 4),
            "Predicted Class": unseen_preds,
        }
    )

    # Export to CSV without indices
    output_filename = f"{user_full_name}_Submission.csv"
    submission_df.to_csv(output_filename, index=False)

    print(f"\nSubmission exported successfully to: {output_filename}")
    print(f"Total Unseen Providers Processed: {len(submission_df)}")
    print(
        f"Predicted Fraudulent Providers: {unseen_preds.sum()} ({unseen_preds.mean()*100:.2f}%)\n"
    )
    print("=== First 10 Rows of Output ===")
    print(submission_df.head(10))


# Execute script (Replace 'Your Name' with your name)
if __name__ == "__main__":
    generate_unseen_predictions(user_full_name="Your_Full_Name")