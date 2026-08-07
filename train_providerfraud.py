import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def run_training_pipeline(X: pd.DataFrame, y: pd.Series):
    """Executes train-test splitting, cross-validation, SMOTE oversampling,

    model training (Baseline Logistic Regression, Random Forest, XGBoost),
    evaluation on validation set, confusion matrix/ROC plotting, and feature importance.
    """
    # 1. Train / Validation Split (Stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("=== Dataset Splitting Summary ===")
    print(
        f"Training set:   {X_train.shape[0]} samples (Fraud: {y_train.sum()})"
    )
    print(
        f"Validation set: {X_val.shape[0]} samples (Fraud: {y_val.sum()})\n"
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle Class Imbalance using SMOTE on Training Set 
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_train_smote_scaled, _ = smote.fit_resample(X_train_scaled, y_train)

    print("=== SMOTE Resampling Summary ===")
    print(f"Original Train Balance: {dict(pd.Series(y_train).value_counts())}")
    print(f"Resampled Train Balance: {dict(pd.Series(y_train_smote).value_counts())}\n")


    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    print("=== Performing 5-Fold Stratified Cross-Validation ===")
    for name, model in models.items():
        cv_roc_scores = []
        cv_pr_scores = []
        cv_f1_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = (
                X_train.iloc[train_idx],
                X_train.iloc[val_idx],
            )
            y_fold_train, y_fold_val = (
                y_train.iloc[train_idx],
                y_train.iloc[val_idx],
            )

            if "Logistic" in name:
              
                fold_scaler = StandardScaler()
                X_fold_train = fold_scaler.fit_transform(X_fold_train)
                X_fold_val = fold_scaler.transform(X_fold_val)

            
            sm = SMOTE(random_state=42)
            X_fold_train_res, y_fold_train_res = sm.fit_resample(
                X_fold_train, y_fold_train
            )

            model.fit(X_fold_train_res, y_fold_train_res)
            y_fold_probs = model.predict_proba(X_fold_val)[:, 1]
            y_fold_preds = model.predict(X_fold_val)

            cv_roc_scores.append(roc_auc_score(y_fold_val, y_fold_probs))
            cv_pr_scores.append(
                average_precision_score(y_fold_val, y_fold_probs)
            )
            cv_f1_scores.append(f1_score(y_fold_val, y_fold_preds))

        cv_summary[name] = {
            "Mean CV ROC-AUC": np.mean(cv_roc_scores),
            "Mean CV PR-AUC": np.mean(cv_pr_scores),
            "Mean CV F1-Score": np.mean(cv_f1_scores),
        }

    print(pd.DataFrame(cv_summary).T.round(4))
    print("\n")

    #  Fit Models on Full Training Set and Evaluate on Validation Set
    val_metrics = {}
    fitted_models = {}
    val_predictions = {}

    for name, model in models.items():
        if "Logistic" in name:
            model.fit(X_train_smote_scaled, y_train_smote)
            probs = model.predict_proba(X_val_scaled)[:, 1]
            preds = model.predict(X_val_scaled)
        else:
            model.fit(X_train_smote, y_train_smote)
            probs = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)

        fitted_models[name] = model
        val_predictions[name] = (preds, probs)

        val_metrics[name] = {
            "ROC-AUC": roc_auc_score(y_val, probs),
            "PR-AUC": average_precision_score(y_val, probs),
            "F1-Score": f1_score(y_val, preds),
        }

    val_metrics_df = pd.DataFrame(val_metrics).T
    print("=== Validation Set Evaluation Metrics ===")
    print(val_metrics_df.round(4))
    print("\n")

    #  Plot ROC Curves and Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ROC Curves
    for name, (preds, probs) in val_predictions.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc_score = val_metrics[name]["ROC-AUC"]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", label="Random Chance")
    axes[0].set_title("Validation ROC Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

   
    best_model_name = val_metrics_df["ROC-AUC"].idxmax()
    best_preds, _ = val_predictions[best_model_name]

    # Plot Confusion Matrix for Best Model
    cm = confusion_matrix(y_val, best_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        cbar=False,
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"],
    )
    axes[1].set_title(
        f"Confusion Matrix: {best_model_name}", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.show()

   
    best_model = fitted_models[best_model_name]
    print(f"=== Top 10 Feature Importances ({best_model_name}) ===")

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print(importances.head(10).round(4))
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(
            np.abs(best_model.coef_[0]), index=X.columns
        ).sort_values(ascending=False)
        print(importances.head(10).round(4))


