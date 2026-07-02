"""
NEXUS Smart City — ML Severity Classifier
Trains on synthetic IoT sensor data to classify:
  - Problem Type  (water_leak, pipe_burst, power_fault, road_damage, drain_overflow, sewage_backup)
  - Severity      (moderate, serious, critical)
  - Department    (water, electricity, municipality, drainage)
"""

import numpy as np
import pickle, os, json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── Sensor feature definitions ────────────────────────────────
SENSOR_FEATURES = [
    "water_pressure",     # PSI  — normal: 40-80
    "water_flow",         # L/min — normal: 80-200
    "pipe_vibration",     # mm/s — normal: 0-2
    "voltage_level",      # V    — normal: 215-245
    "transformer_temp",   # °C   — normal: 20-65
    "current_draw",       # A    — normal: 10-80
    "road_crack_pct",     # %    — normal: 0-10
    "pothole_depth_cm",   # cm   — normal: 0-2
    "drain_flow",         # m³/h — normal: 50-200
    "drain_blockage_pct", # %    — normal: 0-25
    "sewage_level_cm",    # cm   — normal: 0-40
    "zone_rainfall_mm",   # mm/hr — normal: 0-20
]

SENSOR_NORMALS = {
    "water_pressure":     (40,  80),
    "water_flow":         (80,  200),
    "pipe_vibration":     (0,   2),
    "voltage_level":      (215, 245),
    "transformer_temp":   (20,  65),
    "current_draw":       (10,  80),
    "road_crack_pct":     (0,   10),
    "pothole_depth_cm":   (0,   2),
    "drain_flow":         (50,  200),
    "drain_blockage_pct": (0,   25),
    "sewage_level_cm":    (0,   40),
    "zone_rainfall_mm":   (0,   20),
}

SENSOR_WARN = {
    "water_pressure":     (20,  100),
    "water_flow":         (20,  280),
    "pipe_vibration":     (2,   10),
    "voltage_level":      (195, 265),
    "transformer_temp":   (65,  95),
    "current_draw":       (80,  130),
    "road_crack_pct":     (10,  40),
    "pothole_depth_cm":   (2,   12),
    "drain_flow":         (200, 320),
    "drain_blockage_pct": (25,  80),
    "sewage_level_cm":    (40,  90),
    "zone_rainfall_mm":   (20,  60),
}

# ── Problem definitions ───────────────────────────────────────
PROBLEM_PROFILES = {
    "water_leak": {
        "dept": "water",
        "sensors": {"water_pressure": "low", "water_flow": "high", "pipe_vibration": "normal"},
        "severity_thresholds": {"moderate": 0.3, "serious": 0.6, "critical": 0.85},
    },
    "pipe_burst": {
        "dept": "water",
        "sensors": {"water_pressure": "very_low", "water_flow": "very_high", "pipe_vibration": "high"},
        "severity_thresholds": {"moderate": 0.5, "serious": 0.7, "critical": 0.9},
    },
    "water_outage": {
        "dept": "water",
        "sensors": {"water_flow": "zero", "water_pressure": "very_low"},
        "severity_thresholds": {"moderate": 0.4, "serious": 0.65, "critical": 0.88},
    },
    "power_fluctuation": {
        "dept": "electricity",
        "sensors": {"voltage_level": "unstable", "current_draw": "high"},
        "severity_thresholds": {"moderate": 0.35, "serious": 0.6, "critical": 0.8},
    },
    "transformer_overload": {
        "dept": "electricity",
        "sensors": {"transformer_temp": "high", "current_draw": "very_high"},
        "severity_thresholds": {"moderate": 0.45, "serious": 0.7, "critical": 0.9},
    },
    "grid_fault": {
        "dept": "electricity",
        "sensors": {"voltage_level": "very_low", "current_draw": "zero"},
        "severity_thresholds": {"moderate": 0.5, "serious": 0.75, "critical": 0.92},
    },
    "road_damage": {
        "dept": "municipality",
        "sensors": {"road_crack_pct": "high", "pothole_depth_cm": "moderate"},
        "severity_thresholds": {"moderate": 0.3, "serious": 0.55, "critical": 0.8},
    },
    "pothole_formation": {
        "dept": "municipality",
        "sensors": {"pothole_depth_cm": "high", "road_crack_pct": "moderate"},
        "severity_thresholds": {"moderate": 0.35, "serious": 0.6, "critical": 0.82},
    },
    "drain_overflow": {
        "dept": "drainage",
        "sensors": {"drain_blockage_pct": "high", "drain_flow": "very_high", "zone_rainfall_mm": "high"},
        "severity_thresholds": {"moderate": 0.4, "serious": 0.65, "critical": 0.88},
    },
    "sewage_backup": {
        "dept": "drainage",
        "sensors": {"sewage_level_cm": "high", "drain_blockage_pct": "high"},
        "severity_thresholds": {"moderate": 0.45, "serious": 0.68, "critical": 0.9},
    },
    "drain_blockage": {
        "dept": "drainage",
        "sensors": {"drain_blockage_pct": "very_high", "drain_flow": "low"},
        "severity_thresholds": {"moderate": 0.35, "serious": 0.62, "critical": 0.85},
    },
}


def _rng_in_range(lo, hi, n=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(lo, hi, n)


def generate_training_data(n_samples=8000, seed=42):
    """
    Synthesise sensor readings labelled by problem_type and severity.
    Normal samples + anomalous samples per problem profile.
    """
    rng = np.random.default_rng(seed)
    X, y_prob, y_sev, y_dept = [], [], [], []

    feat = SENSOR_FEATURES
    n_feat = len(feat)

    def normal_sample():
        row = []
        for f in feat:
            lo, hi = SENSOR_NORMALS[f]
            row.append(float(rng.uniform(lo, hi)))
        return row

    def anomaly_sample(problem_key, severity_level):
        """Build a sensor vector that matches a problem signature."""
        row = normal_sample()
        profile = PROBLEM_PROFILES[problem_key]
        sev_score = {"moderate": 0.4, "serious": 0.7, "critical": 0.95}[severity_level]

        for i, f in enumerate(feat):
            n_lo, n_hi = SENSOR_NORMALS[f]
            w_lo, w_hi = SENSOR_WARN[f]
            n_mid = (n_lo + n_hi) / 2

            sensor_role = profile["sensors"].get(f)
            if sensor_role is None:
                # Small random noise around normal
                row[i] = float(rng.normal(n_mid, (n_hi - n_lo) * 0.05))
                continue

            if sensor_role == "low":
                target = n_lo - (n_lo - w_lo) * sev_score
            elif sensor_role == "very_low":
                target = w_lo * (1 - sev_score * 0.5)
            elif sensor_role == "zero":
                target = w_lo * 0.05
            elif sensor_role == "high":
                target = n_hi + (w_hi - n_hi) * sev_score
            elif sensor_role == "very_high":
                target = n_hi + (w_hi - n_hi) * (0.5 + sev_score * 0.5)
            elif sensor_role == "moderate":
                target = n_hi + (w_hi - n_hi) * (sev_score * 0.5)
            elif sensor_role == "unstable":
                target = n_mid + rng.choice([-1, 1]) * (n_hi - n_lo) * (0.4 + sev_score * 0.6)
            else:
                target = n_mid

            noise = rng.normal(0, abs(target) * 0.04 + 0.01)
            row[i] = float(np.clip(target + noise, w_lo * 0.5, w_hi * 1.2))

        return row

    # 1. Normal samples (labelled "normal")
    n_normal = n_samples // 5
    for _ in range(n_normal):
        X.append(normal_sample())
        y_prob.append("normal")
        y_sev.append("normal")
        y_dept.append("none")

    # 2. Problem samples
    problems = list(PROBLEM_PROFILES.keys())
    severities = ["moderate", "serious", "critical"]
    per_combo = (n_samples - n_normal) // (len(problems) * len(severities))

    for prob in problems:
        dept = PROBLEM_PROFILES[prob]["dept"]
        for sev in severities:
            for _ in range(per_combo):
                X.append(anomaly_sample(prob, sev))
                y_prob.append(prob)
                y_sev.append(sev)
                y_dept.append(dept)

    return np.array(X), np.array(y_prob), np.array(y_sev), np.array(y_dept)


def train_models(save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    print("🔧 Generating synthetic IoT training data...")
    X, y_prob, y_sev, y_dept = generate_training_data(n_samples=8000)

    print(f"   Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Problem classes: {np.unique(y_prob)}")

    # ── Model 1: Problem Type Classifier ─────────────────────
    print("\n🤖 Training Problem Type Classifier (RandomForest)...")
    le_prob = LabelEncoder().fit(y_prob)
    Xtr, Xte, ytr, yte = train_test_split(X, le_prob.transform(y_prob), test_size=0.2, random_state=42)
    clf_prob = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)),
    ])
    clf_prob.fit(Xtr, ytr)
    prob_acc = accuracy_score(yte, clf_prob.predict(Xte))
    print(f"   Accuracy: {prob_acc*100:.1f}%")

    # ── Model 2: Severity Classifier ─────────────────────────
    print("\n🤖 Training Severity Classifier (GradientBoosting)...")
    le_sev = LabelEncoder().fit(y_sev)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, le_sev.transform(y_sev), test_size=0.2, random_state=42)
    clf_sev = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ])
    clf_sev.fit(Xtr2, ytr2)
    sev_acc = accuracy_score(yte2, clf_sev.predict(Xte2))
    print(f"   Accuracy: {sev_acc*100:.1f}%")

    # ── Model 3: Department Classifier ───────────────────────
    print("\n🤖 Training Department Classifier (RandomForest)...")
    le_dept = LabelEncoder().fit(y_dept)
    Xtr3, Xte3, ytr3, yte3 = train_test_split(X, le_dept.transform(y_dept), test_size=0.2, random_state=42)
    clf_dept = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)),
    ])
    clf_dept.fit(Xtr3, ytr3)
    dept_acc = accuracy_score(yte3, clf_dept.predict(Xte3))
    print(f"   Accuracy: {dept_acc*100:.1f}%")

    # ── Save models ───────────────────────────────────────────
    bundle = {
        "clf_prob": clf_prob, "le_prob": le_prob,
        "clf_sev":  clf_sev,  "le_sev":  le_sev,
        "clf_dept": clf_dept, "le_dept": le_dept,
        "features": SENSOR_FEATURES,
        "accuracy": {
            "problem_type": round(prob_acc, 4),
            "severity":     round(sev_acc, 4),
            "department":   round(dept_acc, 4),
        },
    }
    model_path = os.path.join(save_dir, "nexus_ml_bundle.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    meta = {
        "model_version": "3.2",
        "features": SENSOR_FEATURES,
        "problem_classes": list(le_prob.classes_),
        "severity_classes": list(le_sev.classes_),
        "dept_classes": list(le_dept.classes_),
        "accuracy": bundle["accuracy"],
        "n_training_samples": int(X.shape[0]),
    }
    with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Models saved to {model_path}")
    print(f"   Problem type accuracy : {prob_acc*100:.1f}%")
    print(f"   Severity accuracy     : {sev_acc*100:.1f}%")
    print(f"   Department accuracy   : {dept_acc*100:.1f}%")
    return bundle


def load_models(save_dir="models"):
    path = os.path.join(save_dir, "nexus_ml_bundle.pkl")
    if not os.path.exists(path):
        print("No saved model found — training now...")
        return train_models(save_dir)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle, sensor_values: dict):
    """
    Given a dict of {sensor_name: float_value}, return prediction.
    Missing sensors filled with normal midpoint.
    """
    features = bundle["features"]
    row = []
    for f in features:
        if f in sensor_values:
            row.append(float(sensor_values[f]))
        else:
            lo, hi = SENSOR_NORMALS[f]
            row.append((lo + hi) / 2.0)

    X = np.array(row).reshape(1, -1)
    prob_label = bundle["le_prob"].inverse_transform(bundle["clf_prob"].predict(X))[0]
    sev_label  = bundle["le_sev"].inverse_transform(bundle["clf_sev"].predict(X))[0]
    dept_label = bundle["le_dept"].inverse_transform(bundle["clf_dept"].predict(X))[0]

    prob_proba = bundle["clf_prob"].predict_proba(X)[0]
    sev_proba  = bundle["clf_sev"].predict_proba(X)[0]
    confidence = round(float(np.max(prob_proba) * 0.6 + np.max(sev_proba) * 0.4) * 100, 1)

    return {
        "problem_type": prob_label,
        "severity":     sev_label,
        "department":   dept_label,
        "confidence":   confidence,
        "is_anomaly":   prob_label != "normal",
    }


if __name__ == "__main__":
    train_models("../models")
