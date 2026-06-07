from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path("earthquake_data.csv")
MODEL_PATH = Path("tsunami_tuned_random_forest.joblib")
METADATA_PATH = Path("model_metadata.json")
TARGET = "tsunami"

NUMERIC_FEATURES = [
    "magnitude", "cdi", "mmi", "sig", "nst", "dmin", "gap",
    "depth", "latitude", "longitude"
]

CATEGORICAL_FEATURES = ["alert", "net", "magType", "continent", "country"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required feature columns exist and have consistent types."""
    data = df.copy()

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = data[col].astype("object")

    return data


def load_training_data(data_path: Path = DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.drop_duplicates().reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    df = prepare_dataframe(df)

    X = df[FEATURES]
    y = df[TARGET]

    return X, y, df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_and_save(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH):
    X, y, df = load_training_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    eval_pipeline = build_pipeline()
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }

    # Fit final model on all available data for deployment inference.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    bundle = {
        "model": final_pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "holdout_metrics": metrics,
    }

    joblib.dump(bundle, model_path)

    metadata = {
        "project_title": "Prediction of Tsunami Potential Based on Earthquake Characteristics",
        "model_name": "Tuned Random Forest",
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "holdout_metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "disclaimer": "Academic prototype only. Not a substitute for official tsunami early warning systems.",
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return bundle, metadata


if __name__ == "__main__":
    _, metadata = train_and_save()
    print("Model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)
    print(json.dumps(metadata["holdout_metrics"], indent=2))
