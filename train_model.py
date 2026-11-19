import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FEATURES = [
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "missed_deadlines",
    "timeouts",
]

STATE_LABELS = {
    0: "Normal",
    1: "Degraded",
    2: "Critical Fault",
}


def generate_labelled_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """Generates balanced synthetic ECU communication metrics across 3 current states."""
    rng = np.random.default_rng(random_state)
    per_class = n_samples // 3

    # Class 0: Normal communication
    normal = pd.DataFrame({
        "latency_ms": rng.normal(12.0, 3.0, per_class).clip(2.0, 25.0),
        "jitter_ms": rng.normal(1.5, 0.7, per_class).clip(0.1, 4.5),
        "packet_loss_pct": rng.exponential(0.2, per_class).clip(0.0, 1.5),
        "missed_deadlines": rng.choice([0, 1], size=per_class, p=[0.95, 0.05]),
        "timeouts": np.zeros(per_class, dtype=int),
        "state": 0,
    })

    # Class 1: Degraded communication (bus congestion / Wi-Fi interference)
    degraded = pd.DataFrame({
        "latency_ms": rng.normal(48.0, 12.0, per_class).clip(22.0, 85.0),
        "jitter_ms": rng.normal(9.5, 3.0, per_class).clip(4.0, 20.0),
        "packet_loss_pct": rng.uniform(1.5, 8.0, per_class),
        "missed_deadlines": rng.integers(1, 5, size=per_class),
        "timeouts": rng.choice([0, 1], size=per_class, p=[0.75, 0.25]),
        "state": 1,
    })

    # Class 2: Critical Fault (severe drops / gateway timeout)
    fault = pd.DataFrame({
        "latency_ms": rng.normal(135.0, 30.0, per_class).clip(80.0, 250.0),
        "jitter_ms": rng.normal(28.0, 8.0, per_class).clip(16.0, 60.0),
        "packet_loss_pct": rng.uniform(8.0, 35.0, per_class),
        "missed_deadlines": rng.integers(4, 15, size=per_class),
        "timeouts": rng.integers(1, 8, size=per_class),
        "state": 2,
    })

    df = pd.concat([normal, degraded, fault], ignore_index=True)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_and_save():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = generate_labelled_dataset(n_samples=3000)
    dataset_path = os.path.join("data", "ecu_comm_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Saved labelled dataset to {dataset_path} ({len(df)} rows)")

    X = df[FEATURES]
    y = df["state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[INFO] Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=list(STATE_LABELS.values())))

    model_path = os.path.join("models", "ecu_rf_classifier.joblib")
    joblib.dump(clf, model_path)

    meta = {
        "features": FEATURES,
        "state_labels": STATE_LABELS,
        "validation_accuracy": round(float(acc), 4),
        "feature_importances": dict(
            zip(FEATURES, [round(float(v), 4) for v in clf.feature_importances_])
        ),
    }
    with open(os.path.join("models", "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Saved trained model and metadata to models/")


if __name__ == "__main__":
    train_and_save()