import time
import random
import requests
from typing import Dict, Any, Tuple

DEFAULT_NODERED_URL = "http://localhost:1880/api/vehicle-data"


def normalize_payload(raw: Dict[str, Any]) -> Dict[str, float]:
    """Sanitizes incoming JSON from Node-RED whether keys are flat or nested under 'latest'."""
    data = raw.get("latest", raw) if isinstance(raw, dict) else {}

    def get_num(keys, default=0.0):
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    return float(data[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        # Vehicle subsystem telemetry
        "battery_voltage": round(get_num(["battery_voltage", "batteryVoltage", "voltage"], 12.6), 2),
        "battery_current": round(get_num(["battery_current", "batteryCurrent", "current"], 14.2), 2),
        "temperature": round(get_num(["temperature", "temp", "engine_temp"], 82.0), 1),
        "vehicle_speed": round(get_num(["vehicle_speed", "vehicleSpeed", "speed"], 65.0), 1),
        "soc": round(get_num(["soc", "state_of_charge", "stateOfCharge"], 78.0), 1),
        # ECU communication metrics (AI inputs)
        "latency_ms": round(get_num(["latency_ms", "latency"], 12.0), 2),
        "jitter_ms": round(get_num(["jitter_ms", "jitter"], 1.5), 2),
        "packet_loss_pct": round(get_num(["packet_loss_pct", "packet_loss", "packetLoss"], 0.1), 2),
        "missed_deadlines": int(get_num(["missed_deadlines", "missedDeadlines"], 0)),
        "timeouts": int(get_num(["timeouts"], 0)),
    }


def fetch_nodered_telemetry(
    url: str = DEFAULT_NODERED_URL, timeout: float = 1.2
) -> Tuple[bool, Dict[str, float], str]:
    """Polls Node-RED HTTP API. Returns (success, normalized_data, status_message)."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return True, normalize_payload(payload), "Connected to Node-RED"
    except Exception as exc:
        return False, {}, f"Node-RED unreachable ({exc.__class__.__name__})"


def generate_simulated_telemetry(mode: str = "Normal") -> Dict[str, float]:
    """Generates local test telemetry for testing AI states without Node-RED."""
    if mode == "Degraded":
        comm = {
            "latency_ms": round(random.uniform(38.0, 68.0), 2),
            "jitter_ms": round(random.uniform(7.0, 15.0), 2),
            "packet_loss_pct": round(random.uniform(2.0, 6.5), 2),
            "missed_deadlines": random.randint(1, 4),
            "timeouts": random.choice([0, 1]),
        }
    elif mode == "Critical Fault":
        comm = {
            "latency_ms": round(random.uniform(110.0, 195.0), 2),
            "jitter_ms": round(random.uniform(22.0, 42.0), 2),
            "packet_loss_pct": round(random.uniform(10.0, 26.0), 2),
            "missed_deadlines": random.randint(5, 12),
            "timeouts": random.randint(2, 6),
        }
    else:
        comm = {
            "latency_ms": round(random.uniform(8.0, 18.0), 2),
            "jitter_ms": round(random.uniform(0.6, 2.8), 2),
            "packet_loss_pct": round(random.uniform(0.0, 0.6), 2),
            "missed_deadlines": 0,
            "timeouts": 0,
        }

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "battery_voltage": round(random.uniform(12.3, 13.8), 2),
        "battery_current": round(random.uniform(8.0, 22.0), 2),
        "temperature": round(random.uniform(78.0, 92.0), 1),
        "vehicle_speed": round(random.uniform(45.0, 88.0), 1),
        "soc": round(random.uniform(72.0, 85.0), 1),
        **comm,
    }