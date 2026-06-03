from __future__ import annotations

import argparse
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_XLS = RAW_DIR / "default of credit card clients.xls"


RENAME_MAP = {
    "ID": "client_id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt_1",
    "BILL_AMT2": "bill_amt_2",
    "BILL_AMT3": "bill_amt_3",
    "BILL_AMT4": "bill_amt_4",
    "BILL_AMT5": "bill_amt_5",
    "BILL_AMT6": "bill_amt_6",
    "PAY_AMT1": "pay_amt_1",
    "PAY_AMT2": "pay_amt_2",
    "PAY_AMT3": "pay_amt_3",
    "PAY_AMT4": "pay_amt_4",
    "PAY_AMT5": "pay_amt_5",
    "PAY_AMT6": "pay_amt_6",
    "default payment next month": "default_next_month",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLS.exists() and not force:
        return RAW_XLS

    zip_path = RAW_DIR / "uci_default_credit_card_clients.zip"
    print(f"Downloading UCI dataset to {zip_path} ...")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        xls_names = [name for name in zf.namelist() if name.lower().endswith(".xls")]
        if not xls_names:
            raise FileNotFoundError("No .xls file found inside the UCI dataset zip.")
        zf.extract(xls_names[0], RAW_DIR)
        extracted = RAW_DIR / xls_names[0]
        if extracted != RAW_XLS:
            extracted.replace(RAW_XLS)

    return RAW_XLS


def load_data(raw_path: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_path, header=1)
    df = df.rename(columns=RENAME_MAP)
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})

    pay_cols = [f"pay_{i}" for i in range(1, 7)]
    bill_cols = [f"bill_amt_{i}" for i in range(1, 7)]
    pay_amt_cols = [f"pay_amt_{i}" for i in range(1, 7)]

    df["utilization_recent"] = safe_divide(df["bill_amt_1"], df["limit_bal"])
    df["payment_ratio_recent"] = safe_divide(df["pay_amt_1"], df["bill_amt_1"])
    df["avg_delay"] = df[pay_cols].mean(axis=1)
    df["max_delay"] = df[pay_cols].max(axis=1)
    df["delinquency_count"] = (df[pay_cols] > 0).sum(axis=1)
    df["severe_delay_count"] = (df[pay_cols] >= 2).sum(axis=1)
    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)
    df["bill_growth"] = safe_divide(df["bill_amt_1"] - df["bill_amt_6"], df["bill_amt_6"].abs())
    df["payment_to_bill_6m"] = safe_divide(df[pay_amt_cols].sum(axis=1), df[bill_cols].sum(axis=1))

    df["age_group"] = pd.cut(
        df["age"],
        bins=[20, 29, 39, 49, 59, math.inf],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype(str)
    df["limit_band"] = pd.cut(
        df["limit_bal"],
        bins=[0, 50000, 100000, 200000, 500000, math.inf],
        labels=["<=50K", "50K-100K", "100K-200K", "200K-500K", "500K+"],
        include_lowest=True,
    ).astype(str)
    df["utilization_band"] = pd.cut(
        df["utilization_recent"],
        bins=[-math.inf, 0.25, 0.50, 0.75, 1.00, math.inf],
        labels=["<=25%", "25%-50%", "50%-75%", "75%-100%", "100%+"],
    ).astype(str)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def model_confusion_rows(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, int | str]]:
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    rows = []
    for actual_idx, actual_label in enumerate([0, 1]):
        for pred_idx, pred_label in enumerate([0, 1]):
            rows.append(
                {
                    "model": name,
                    "actual": actual_label,
                    "predicted": pred_label,
                    "count": int(matrix[actual_idx, pred_idx]),
                }
            )
    return rows


def risk_tier(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.40:
        return "Medium"
    return "High"


def train_and_export(force_download: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = download_dataset(force=force_download)
    df = clean_and_engineer_features(load_data(raw_path))

    target = "default_next_month"
    exclude = {target, "client_id"}
    categorical_features = ["sex", "education", "marriage", "age_group", "limit_band", "utilization_band"]
    numeric_features = [col for col in df.columns if col not in exclude and col not in categorical_features]

    x = df[numeric_features + categorical_features]
    y = df[target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    non_default = int((y_train == 0).sum())
    default = int((y_train == 1).sum())
    scale_pos_weight = non_default / default

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42, cache_size=1000),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    fitted_models: dict[str, Pipeline] = {}
    metric_rows = []
    confusion_rows = []

    for name, estimator in models.items():
        print(f"Training {name} ...")
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline
        metric_rows.append(evaluate_model(name, pipeline, x_test, y_test))
        confusion_rows.extend(model_confusion_rows(name, pipeline, x_test, y_test))

    final_model = fitted_models["XGBoost"]
    df["pred_default_prob"] = final_model.predict_proba(x)[:, 1]
    df["risk_tier"] = df["pred_default_prob"].apply(risk_tier)
    df["recommended_action"] = np.select(
        [
            df["risk_tier"].eq("High"),
            df["risk_tier"].eq("Medium"),
        ],
        [
            "Monitor closely / review limit",
            "Watch repayment behavior",
        ],
        default="Maintain standard monitoring",
    )

    model_metrics = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    confusion = pd.DataFrame(confusion_rows)

    xgb_model = final_model.named_steps["model"]
    preprocessor_fitted = final_model.named_steps["preprocess"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": xgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    scored_path = PROCESSED_DIR / "credit_default_scored.csv"
    metrics_path = PROCESSED_DIR / "model_metrics.csv"
    confusion_path = PROCESSED_DIR / "confusion_matrix.csv"
    feature_path = PROCESSED_DIR / "feature_importance.csv"

    df.to_csv(scored_path, index=False)
    model_metrics.to_csv(metrics_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    feature_importance.to_csv(feature_path, index=False)

    summary = {
        "rows": int(len(df)),
        "default_rate": float(df[target].mean()),
        "final_model": "XGBoost",
        "best_model_by_roc_auc": str(model_metrics.iloc[0]["model"]),
        "outputs": {
            "scored": str(scored_path),
            "metrics": str(metrics_path),
            "confusion": str(confusion_path),
            "feature_importance": str(feature_path),
        },
    }
    (PROCESSED_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit default risk models and export Excel-ready CSV files.")
    parser.add_argument("--force-download", action="store_true", help="Download the UCI dataset again.")
    args = parser.parse_args()
    train_and_export(force_download=args.force_download)


if __name__ == "__main__":
    main()



