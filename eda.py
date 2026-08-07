import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def perform_eda_and_feature_selection(df: pd.DataFrame):
    """Performs Exploratory Data Analysis (EDA), visualizes key provider metrics,

    removes collinear features (> 0.85 correlation), and returns X, y.
    """
    # 1. Class Distribution Analysis
    class_counts = df["PotentialFraud"].value_counts()
    fraud_pct = df["PotentialFraud"].mean() * 100

    print("=== Class Distribution ===")
    print(f"Non-Fraudulent Providers (0): {class_counts.get(0, 0)}")
    print(f"Fraudulent Providers (1):     {class_counts.get(1, 0)}")
    print(f"Fraud Percentage:             {fraud_pct:.2f}%\n")

    # Standardize column naming if necessary
    if "inpatient_ratio" not in df.columns:
        df["inpatient_ratio"] = df["inpatient_claims"] / df["total_claims"]

    metrics = [
        ("mean_reimbursement", "Avg Reimbursement ($)"),
        ("total_claims", "Total Claims Count"),
        ("inpatient_ratio", "Inpatient Claim Ratio"),
        ("unique_patients", "Unique Patients Served"),
    ]

    # 2. Summary Statistics Comparison
    metric_cols = [m[0] for m in metrics]
    summary_stats = df.groupby("PotentialFraud")[metric_cols].agg(
        ["mean", "median", "std"]
    )
    print("=== Summary Statistics by Fraud Status ===")
    print(summary_stats.T.round(2))
    print("\n")

    # 3. Box Plots Generation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        sns.boxplot(
            x="PotentialFraud",
            y=col,
            data=df,
            ax=axes[idx],
            palette={0: "#2ecc71", 1: "#e74c3c"},
            showfliers=False,  # Exclude extreme outliers for visualization clarity
        )
        axes[idx].set_title(
            f"{title} by Fraud Status", fontsize=12, fontweight="bold"
        )
        axes[idx].set_xlabel("Potential Fraud (0 = No, 1 = Yes)")
        axes[idx].set_ylabel(title)

    plt.tight_layout()
    plt.show()

    # 4. Multicollinearity Identification & Feature Dropping (> 0.85 Correlation)
    numeric_df = df.drop(
        columns=["Provider", "PotentialFraud"], errors="ignore"
    )

    # Calculate absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than 0.85
    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > 0.85)
    ]

    print("=== Multicollinearity Analysis (Threshold > 0.85) ===")
    print(
        f"Identified {len(to_drop)} highly collinear features to drop: {to_drop}\n"
    )

    # Drop collinear features
    df_reduced = df.drop(columns=to_drop)

    # 5. Separate into Features (X) and Target (y)
    X = df_reduced.drop(columns=["Provider", "PotentialFraud"], errors="ignore")
    y = df_reduced["PotentialFraud"]

    # Impute remaining missing values (e.g., avg_length_of_stay / inpatient_outpatient_ratio) with 0
    X = X.fillna(0)

    print("=== Final Dataset Shapes ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape:   {y.shape}")

    return X, y, summary_stats


# Example Usage (assuming 'final_df' is loaded from step 1):
# X, y, summary_stats = perform_eda_and_feature_selection(final_df)