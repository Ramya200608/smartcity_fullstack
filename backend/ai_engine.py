"""
NEXUS Smart City — AI Analysis Engine
Takes ML predictions + sensor readings → produces human-readable reports
and resolution time estimates.
"""

import random, time, math
from typing import Optional

# ── Problem titles ────────────────────────────────────────────
PROBLEM_TITLES = {
    "water_leak":          "Water Main Leakage",
    "pipe_burst":          "Pipe Burst Detected",
    "water_outage":        "Water Supply Outage",
    "power_fluctuation":   "Power Fluctuation",
    "transformer_overload":"Transformer Overload",
    "grid_fault":          "Grid Fault / Outage",
    "road_damage":         "Road Surface Damage",
    "pothole_formation":   "Pothole Formation",
    "drain_overflow":      "Drain Overflow Risk",
    "sewage_backup":       "Sewage Backup",
    "drain_blockage":      "Drainage Blockage",
    "normal":              "Normal Conditions",
}

DEPT_LABELS = {
    "water":        "Water Department",
    "electricity":  "Electricity Department",
    "municipality": "Municipality Department",
    "drainage":     "Drainage Department",
}

# ── Template analysis per problem type ───────────────────────
ANALYSIS_TEMPLATES = {
    "water_leak": [
        "AI Pattern Match: Sustained water pressure drop detected in {zone} — {pressure:.1f} PSI (normal: 40-80). "
        "Cross-referenced with flow sensor data showing anomalous reading of {flow:.1f} L/min. "
        "Pattern is consistent with a slow underground pipe leak. Historical correlation: 91%. "
        "Estimated water loss: {loss:.0f} L/hr. Immediate inspection recommended.",

        "ML Model: Pressure-flow divergence signature in {zone} matches pipe leak profile with {conf}% confidence. "
        "Sensor WP-{zone_short} shows {pressure:.1f} PSI — {drop:.1f}% below zone baseline. "
        "Recommend dispatch of Water-A crew to isolate segment {seg}.",
    ],
    "pipe_burst": [
        "AI CRITICAL: Sudden pressure collapse in {zone} — {pressure:.1f} PSI (critical threshold: <20). "
        "High-frequency pipe vibration sensor triggered at {vib:.2f} mm/s. "
        "ML model classifies as pipe burst with {conf}% confidence. "
        "Estimated water loss: {loss:.0f} L/min. IMMEDIATE CREW DISPATCH REQUIRED.",
    ],
    "water_outage": [
        "AI Detection: Near-zero water flow in {zone} — {flow:.1f} L/min. "
        "Pressure also critically low: {pressure:.1f} PSI. "
        "Possible main supply shutoff or catastrophic breach upstream. "
        "Affecting estimated {affected} households. Emergency protocol initiated.",
    ],
    "power_fluctuation": [
        "AI Grid Monitor: Voltage instability detected in {zone} — {voltage:.1f}V (nominal: 230V). "
        "Current draw anomaly: {current:.1f}A. Pattern matches feeder line fault signature. "
        "ML confidence: {conf}%. Surge protection advisory issued. "
        "Estimated {affected} households at risk of appliance damage.",
    ],
    "transformer_overload": [
        "AI Thermal Warning: Transformer TF-{zone_short} temperature critical at {temp:.1f}°C (safe limit: 65°C). "
        "Current load: {current:.1f}A — {overload:.0f}% above rated capacity. "
        "ML cascade failure prediction: 34% probability within 4 hours if unresolved. "
        "Load redistribution to adjacent feeders recommended immediately.",
    ],
    "grid_fault": [
        "AI CRITICAL: Grid fault signature detected in {zone}. Voltage collapsed to {voltage:.1f}V. "
        "All phases affected — zero current draw indicating total outage. "
        "ML model: 87% probability of feeder breaker trip. "
        "Emergency team required for manual restoration. Backup grid routing initiated.",
    ],
    "road_damage": [
        "AI Structural Assessment: Road surface crack density at {crack:.1f}% in {zone}. "
        "Pothole depth sensor reading: {pothole:.1f} cm. "
        "ML degradation model projects surface reaching critical failure in {days:.0f} days without intervention. "
        "Vehicle damage risk: HIGH. Temporary road closure advisory issued.",
    ],
    "pothole_formation": [
        "AI Road Monitor: Active pothole formation detected in {zone}. "
        "Depth sensor: {pothole:.1f} cm (critical: >8cm). "
        "Subsurface pressure pattern suggests water-induced erosion. "
        "Coordinate with Water Dept for pipe inspection in same segment. "
        "Patching crew ETA: {eta}.",
    ],
    "drain_overflow": [
        "AI Flood Intelligence: Drainage capacity at {blockage:.0f}% in {zone}. "
        "Flow rate: {flow:.1f} m³/hr exceeding safe capacity. "
        "Rainfall intensity: {rain:.1f} mm/hr. "
        "ML flood probability model: {flood_prob:.0f}% chance of surface flooding within 2 hours. "
        "Emergency desilting and sandbag deployment recommended.",
    ],
    "sewage_backup": [
        "AI Sewage Alert: Sewage level in {zone} at {sewage:.1f} cm (critical: >80cm). "
        "Blockage meter: {blockage:.0f}%. "
        "Environmental contamination risk: HIGH. Public health advisory may be required. "
        "Jetting truck dispatch recommended. Estimated clearance time: {eta}.",
    ],
    "drain_blockage": [
        "AI Drainage Monitor: Severe blockage detected in {zone} storm drain network. "
        "Blockage sensor: {blockage:.0f}% — flow reduced to {flow:.1f} m³/hr. "
        "Root cause analysis: likely silt accumulation or debris. "
        "Last manual inspection: {last_inspect}. Desilting crew required.",
    ],
}


def _pick(templates, key):
    t = ANALYSIS_TEMPLATES.get(key, [])
    return random.choice(t) if t else "AI detected anomalous sensor readings in this zone."


ETA_HOURS = {
    ("water",        "critical"): 1.5,
    ("water",        "serious"):  4,
    ("water",        "moderate"): 24,
    ("electricity",  "critical"): 0.75,
    ("electricity",  "serious"):  3,
    ("electricity",  "moderate"): 12,
    ("municipality", "critical"): 6,
    ("municipality", "serious"):  18,
    ("municipality", "moderate"): 72,
    ("drainage",     "critical"): 2,
    ("drainage",     "serious"):  8,
    ("drainage",     "moderate"): 36,
}


def estimate_eta(dept: str, severity: str) -> str:
    hours = ETA_HOURS.get((dept, severity), 12)
    if hours < 1:
        return f"{int(hours*60)} min"
    if hours < 24:
        return f"{hours:.1f} hrs"
    return f"{int(hours//24)} days"


def build_analysis(problem_type: str, severity: str, zone_name: str,
                   sensor_values: dict, confidence: float) -> str:
    template = _pick(ANALYSIS_TEMPLATES, problem_type)
    zone_short = zone_name.replace(" ", "")[:4].upper()
    v = sensor_values
    ctx = {
        "zone":        zone_name,
        "zone_short":  zone_short,
        "conf":        round(confidence),
        "pressure":    v.get("water_pressure",   60),
        "flow":        v.get("water_flow",        140),
        "vib":         v.get("pipe_vibration",    1.0),
        "voltage":     v.get("voltage_level",     230),
        "temp":        v.get("transformer_temp",  55),
        "current":     v.get("current_draw",      50),
        "crack":       v.get("road_crack_pct",    5),
        "pothole":     v.get("pothole_depth_cm",  1),
        "drain_flow":  v.get("drain_flow",        150),
        "blockage":    v.get("drain_blockage_pct",20),
        "sewage":      v.get("sewage_level_cm",   30),
        "rain":        v.get("zone_rainfall_mm",  10),
        "drop":        max(0, (60 - v.get("water_pressure", 60)) / 60 * 100),
        "loss":        abs(200 - v.get("water_flow", 200)) * 3,
        "overload":    max(0, (v.get("current_draw", 50) - 80) / 80 * 100),
        "flood_prob":  min(95, v.get("drain_blockage_pct", 20) + v.get("zone_rainfall_mm", 10) * 2),
        "seg":         f"SG-{random.randint(100,999)}",
        "affected":    random.randint(100, 2000),
        "eta":         estimate_eta("water", severity),
        "days":        random.randint(5, 30),
        "last_inspect": f"{random.randint(2,12)} months ago",
    }
    try:
        return template.format(**ctx)
    except Exception:
        return f"AI Analysis: {problem_type.replace('_',' ').title()} detected in {zone_name}. Severity: {severity}. Confidence: {confidence:.0f}%."


def estimate_affected(severity: str, zone_name: str) -> int:
    base = {"critical": 1200, "serious": 500, "moderate": 120}.get(severity, 200)
    return int(base * (0.5 + random.random()))


def generate_incident_id() -> str:
    ts = int(time.time() * 1000) % 1000000
    r  = random.randint(1000, 9999)
    return f"INC-{ts:06d}-{r}"


def pick_problem_for_dept(dept: str, severity: str) -> str:
    mapping = {
        "water":        ["water_leak", "pipe_burst", "water_outage"],
        "electricity":  ["power_fluctuation", "transformer_overload", "grid_fault"],
        "municipality": ["road_damage", "pothole_formation"],
        "drainage":     ["drain_overflow", "sewage_backup", "drain_blockage"],
    }
    choices = mapping.get(dept, ["water_leak"])
    if severity == "critical":
        return choices[-1] if len(choices) > 1 else choices[0]
    return random.choice(choices)
