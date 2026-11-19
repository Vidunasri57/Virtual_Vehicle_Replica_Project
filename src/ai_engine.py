import os
import json
import joblib
import pandas as pd
from typing import Dict, Any

MODEL_PATH = os.path.join("models", "ecu_rf_classifier.joblib")
META_PATH = os.path.join("models", "model_metadata.json")

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


class ECUCommunicationAI:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Run `python train_model.py` first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.metadata = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                self.metadata = json.load(f)

    def evaluate(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Classifies the current ECU communication state and builds diagnostics."""
        row = pd.DataFrame([{feat: telemetry[feat] for feat in FEATURES}])
        pred_class = int(self.model.predict(row)[0])
        probs = self.model.predict_proba(row)[0]

        # Current degradation severity index (0 = healthy, 100 = severe current fault)
        severity_score = round(float(probs[1] * 50.0 + probs[2] * 100.0), 1)

        diagnostics = []
        if telemetry["latency_ms"] > 30:
            diagnostics.append(f"High bus/Wi-Fi latency ({telemetry['latency_ms']} ms)")
        if telemetry["jitter_ms"] > 5:
            diagnostics.append(f"Elevated timing jitter ({telemetry['jitter_ms']} ms)")
        if telemetry["packet_loss_pct"] > 1.5:
            diagnostics.append(f"Packet drops detected ({telemetry['packet_loss_pct']}%)")
        if telemetry["missed_deadlines"] > 0:
            diagnostics.append(f"{telemetry['missed_deadlines']} missed periodic ECU deadline(s)")
        if telemetry["timeouts"] > 0:
            diagnostics.append(f"{telemetry['timeouts']} gateway/ECU timeout(s)")

        if not diagnostics:
            diagnostics.append("All ECU timing and packet delivery metrics within nominal bounds.")

        return {
            "state_code": pred_class,
            "state_label": STATE_LABELS[pred_class],
            "confidence_pct": round(float(probs[pred_class]) * 100.0, 1),
            "class_probabilities": {
                "Normal": round(float(probs[0]) * 100.0, 1),
                "Degraded": round(float(probs[1]) * 100.0, 1),
                "Critical Fault": round(float(probs[2]) * 100.0, 1),
            },
            "severity_score": severity_score,
            "diagnostics": diagnostics,
        }