import numpy as np
import pandas as pd


def load_datasets(
    train_path: str,
    inpatient_path: str,
    outpatient_path: str,
    beneficiary_path: str,
):
    """Load raw CSV datasets."""
    target_df = pd.read_csv(train_path)
    inpatient_df = pd.read_csv(inpatient_path)
    outpatient_df = pd.read_csv(outpatient_path)
    beneficiary_df = pd.read_csv(beneficiary_path)

    return target_df, inpatient_df, outpatient_df, beneficiary_df


def preprocess_claims_data(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    beneficiary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combines inpatient and outpatient claims, merges beneficiary data,

    and performs feature engineering at the claim/patient level.
    """
    # 1. Add is_inpatient flag
    inpatient_df = inpatient_df.copy()
    outpatient_df = outpatient_df.copy()
    inpatient_df["is_inpatient"] = 1
    outpatient_df["is_inpatient"] = 0

    # Merge Inpatient and Outpatient claim datasets
    claims_df = pd.concat([inpatient_df, outpatient_df], ignore_index=True)

    # Encode chronic conditions: CMS dataset encodes 1=Yes, 2=No -> convert 2 to 0
    chronic_cols = [
        c for c in beneficiary_df.columns if c.startswith("ChronicCond_")
    ]
    beneficiary_df = beneficiary_df.copy()
    beneficiary_df["num_chronic_conditions"] = (
        beneficiary_df[chronic_cols].replace(2, 0).sum(axis=1)
    )

    # 2. Merge combined claim dataset with Beneficiary dataset on BeneID
    claims_bene_df = pd.merge(claims_df, beneficiary_df, on="BeneID", how="left")

    # Convert datetime columns
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

    # Pre-aggregate calculations
    # Average patient age at claim start date
    claims_bene_df["patient_age"] = (
        claims_bene_df["ClaimStartDt"] - claims_bene_df["DOB"]
    ).dt.days / 365.25

    # Inpatient length of stay (DischargeDate - AdmissionDate + 1)
    claims_bene_df["length_of_stay"] = (
        claims_bene_df["DischargeDt"] - claims_bene_df["AdmissionDt"]
    ).dt.days + 1

    # Check if AttendingPhysician == OperatingPhysician
    claims_bene_df["attending_is_operating"] = (
        (claims_bene_df["AttendingPhysician"].notna())
        & (
            claims_bene_df["AttendingPhysician"]
            == claims_bene_df["OperatingPhysician"]
        )
    ).astype(int)

    return claims_bene_df


def aggregate_provider_features(
    claims_bene_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregates claim/patient features up to the Provider level

    and merges with the target label.
    """
    grouped = claims_bene_df.groupby("Provider")

    # 3. Aggregate all claim-level and patient-level metrics
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
        avg_length_of_stay=(
            "length_of_stay",
            "mean",
        ),  # Ignores NaN outpatient records
    ).reset_index()

    # Ratio of Inpatient to Outpatient claims
    provider_agg["outpatient_claims"] = (
        provider_agg["total_claims"] - provider_agg["inpatient_claims"]
    )
    provider_agg["inpatient_outpatient_ratio"] = (
        provider_agg["inpatient_claims"]
        / provider_agg["outpatient_claims"].replace(0, np.nan)
    )

    # Unique physicians combined across attending, operating, and other roles
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

    # 4. Map target label PotentialFraud: 'Yes' -> 1, 'No' -> 0
    target_df = target_df.copy()
    target_df["PotentialFraud"] = target_df["PotentialFraud"].map(
        {"Yes": 1, "No": 0}
    )

    # Merge aggregated table with ground truth dataset
    final_provider_df = pd.merge(
        provider_agg, target_df, on="Provider", how="left"
    )

    return final_provider_df


def main():
    # File paths
    train_file = "Train-1542865627584.csv"
    inpatient_file = "Train_Inpatientdata-1542865627584.csv"
    outpatient_file = "Train_Outpatientdata-1542865627584.csv"
    beneficiary_file = "Train_Beneficiarydata-1542865627584.csv"

    # Execution pipeline
    target_df, inpatient_df, outpatient_df, beneficiary_df = load_datasets(
        train_file, inpatient_file, outpatient_file, beneficiary_file
    )

    claims_bene_df = preprocess_claims_data(
        inpatient_df, outpatient_df, beneficiary_df
    )

    final_df = aggregate_provider_features(claims_bene_df, target_df)

    # Output execution summary
    print("=== Pipeline Summary ===")
    print(f"Aggregated Provider Dataset Shape: {final_df.shape}\n")
    print("=== Missing Values Summary ===")
    print(final_df.isna().sum())


if __name__ == "__main__":
    main()