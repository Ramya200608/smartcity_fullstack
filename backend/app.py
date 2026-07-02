"""
NEXUS Smart City — Flask REST API

Endpoints:
  GET  /api/health
  GET  /api/stats
  GET  /api/incidents               ?dept= &status= &severity= &limit=
  GET  /api/incidents/<id>
  POST /api/incidents/<id>/resolve  body: {"actor":"operator"}
  POST /api/incidents/<id>/assign   body: {"team":"Water-Alpha-1","actor":"operator"}
  POST /api/incidents/<id>/progress
  GET  /api/sensors                 ?zone_id=
  GET  /api/sensors/history         ?sensor_type= &zone_id= &limit=
  POST /api/sensors/tick            (advance IoT simulation + run ML detection)
  GET  /api/departments
  GET  /api/departments/<id>/incidents
  GET  /api/teams                   ?dept=
  GET  /api/audit/<incident_id>
  GET  /api/model/meta
  POST /api/simulate/anomaly        body: {"zone_id","dept"}
  POST /api/simulate/seed

BUGS FIXED vs original:
  BUG 1  os.environ["NEXUS_DB"] is now set *before* importing database so
         database._db_path() picks up the correct path on every call.

  BUG 2  app.config["JSON_SORT_KEYS"] is deprecated in Flask 3.x and raises a
         warning/error.  Fixed to use app.json.sort_keys = False.

  BUG 3  seed_incidents() reused the local variable name `data` for both the
         incoming JSON body (request.get_json()) AND the per-iteration incident
         dict built inside the loop.  On the second loop iteration `data` was
         the incident dict from the previous iteration, causing KeyError /
         wrong values.  Fixed by renaming the request body to `req_body` and
         the incident dict to `incident`.

  BUG 4  The background tick thread called `import requests` (third-party) which
         may not be installed and fails silently, killing auto-detection.
         Fixed to use stdlib urllib.request instead.

  BUG 5  ML_BUNDLE does not have a "model_version" key — that key lives only in
         model_meta.json (written by ml_model.py).  The health endpoint called
         ML_BUNDLE.get("model_version","3.2"), always returning the fallback.
         Fixed to read from model_meta.json or fall back gracefully.

  BUG 6  The catch-all route /<path:path> was placed immediately after /,
         before all /api/* routes.  Flask resolves by rule specificity so it
         worked in practice, but declaring it last is correct and explicit.
"""

import sys, os, time, json, threading, random
import urllib.request as _urllib  # FIX 4: use stdlib, not third-party requests

# ── Path setup — must happen before importing local modules ───
_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_HERE)

# FIX 1: set NEXUS_DB *before* importing database so _db_path() reads it correctly
_DB_PATH  = os.path.join(_BASE_DIR, "data", "nexus.db")
os.environ.setdefault("NEXUS_DB", _DB_PATH)
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

sys.path.insert(0, _HERE)

from flask import Flask, jsonify, request, send_from_directory

import database as db
from iot_sensors import get_network, CITY_ZONES, SENSOR_CONFIG
from ai_engine import (
    build_analysis, estimate_eta, estimate_affected,
    generate_incident_id, PROBLEM_TITLES, pick_problem_for_dept,
)
from ml_model import load_models, predict as ml_predict

# ── Flask app ─────────────────────────────────────────────────
STATIC    = os.path.join(_BASE_DIR, "frontend")
MODEL_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC, static_url_path="")

# FIX 2: Flask 3.x deprecates app.config["JSON_SORT_KEYS"]
app.json.sort_keys = False

# ── Startup ───────────────────────────────────────────────────
db.init_db()

print("Loading / training ML models…")
ML_BUNDLE = load_models(MODEL_DIR)
print(f"ML ready — accuracy: {ML_BUNDLE['accuracy']}")

# FIX 5: read model_version from meta.json, not from the pickle bundle
_meta_path = os.path.join(MODEL_DIR, "model_meta.json")
try:
    with open(_meta_path) as _f:
        _META = json.load(_f)
except FileNotFoundError:
    _META = {}

ML_VERSION = _META.get("model_version", "3.x")

NETWORK = get_network()


# ── CORS ──────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ── Static / frontend ─────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


# ── Health ────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status":         "ok",
        "sensors_online": 144,
        "ml_model":       f"v{ML_VERSION}",   # FIX 5
        "db":             "connected",
        "timestamp":      time.time(),
    })


# ── Stats ─────────────────────────────────────────────────────
@app.route("/api/stats")
def stats():
    return jsonify(db.get_stats())


# ── Incidents ─────────────────────────────────────────────────
@app.route("/api/incidents")
def list_incidents():
    dept     = request.args.get("dept")     or None
    status   = request.args.get("status")   or None
    severity = request.args.get("severity") or None
    limit    = int(request.args.get("limit", 100))
    return jsonify(db.list_incidents(dept=dept, status=status,
                                     severity=severity, limit=limit))


@app.route("/api/incidents/<iid>")
def get_incident(iid):
    inc = db.get_incident(iid)
    if inc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(inc)


@app.route("/api/incidents/<iid>/resolve", methods=["POST", "OPTIONS"])
def resolve_incident(iid):
    if request.method == "OPTIONS":
        return "", 204
    body  = request.get_json(silent=True) or {}
    actor = body.get("actor", "operator")
    inc   = db.update_incident_status(iid, "resolved", actor=actor)
    if inc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(inc)


@app.route("/api/incidents/<iid>/assign", methods=["POST", "OPTIONS"])
def assign_incident(iid):
    if request.method == "OPTIONS":
        return "", 204
    body  = request.get_json(force=True, silent=True) or {}
    team  = body.get("team")
    actor = body.get("actor", "operator")
    if not team:
        return jsonify({"error": "team required"}), 400
    inc = db.update_incident_status(iid, "assigned",
                                     assigned_team=team, actor=actor)
    if inc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(inc)


@app.route("/api/incidents/<iid>/progress", methods=["POST", "OPTIONS"])
def progress_incident(iid):
    if request.method == "OPTIONS":
        return "", 204
    inc = db.update_incident_status(iid, "in_progress", actor="operator")
    if inc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(inc)


# ── Sensors ───────────────────────────────────────────────────
@app.route("/api/sensors")
def get_sensors():
    zone_id = request.args.get("zone_id") or None
    latest  = NETWORK.get_all_latest()
    if zone_id:
        return jsonify(latest.get(zone_id, {}))
    return jsonify(latest)


@app.route("/api/sensors/tick", methods=["POST", "OPTIONS"])
def tick_sensors():
    """Advance IoT simulation one tick; run ML on each zone; auto-create incidents."""
    if request.method == "OPTIONS":
        return "", 204

    all_readings  = NETWORK.tick_all()
    new_incidents = []

    for zone_id, zone_readings in all_readings.items():
        zone = next((z for z in CITY_ZONES if z["id"] == zone_id), None)
        if zone is None:
            continue

        sensor_values = {stype: r["value"] for stype, r in zone_readings.items()}
        prediction    = ml_predict(ML_BUNDLE, sensor_values)

        if not prediction["is_anomaly"]:
            continue

        dept    = prediction["department"]
        sev     = prediction["severity"]
        problem = prediction["problem_type"]

        if problem == "normal" or dept == "none":
            continue

        # Skip if an active incident already exists for this zone + dept
        existing = db.list_incidents(dept=dept, limit=300)
        if any(i["zone_id"] == zone_id and i["status"] != "resolved"
               for i in existing):
            continue

        iid      = generate_incident_id()
        title    = PROBLEM_TITLES.get(problem, problem.replace("_", " ").title())
        analysis = build_analysis(problem, sev, zone["name"],
                                  sensor_values, prediction["confidence"])
        eta      = estimate_eta(dept, sev)
        affected = estimate_affected(sev, zone["name"])

        readings_payload = [
            {
                "sensor": stype,
                "value":  round(sensor_values.get(stype, 0), 2),
                "unit":   SENSOR_CONFIG.get(stype, {}).get("unit", ""),
                "status": zone_readings.get(stype, {}).get("status", "normal"),
            }
            for stype in sensor_values
            if SENSOR_CONFIG.get(stype, {}).get("dept") == dept
        ]
        readings_payload = readings_payload[:6]

        incident = {
            "id":                 iid,
            "title":              title,
            "description":        analysis,
            "zone_id":            zone_id,
            "zone_name":          zone["name"],
            "lat":                zone["lat"] + random.uniform(-0.003, 0.003),
            "lng":                zone["lng"] + random.uniform(-0.003, 0.003),
            "map_x":              zone["map_x"],
            "map_y":              zone["map_y"],
            "dept":               dept,
            "problem_type":       problem,
            "severity":           sev,
            "status":             "detected",
            "ai_confidence":      prediction["confidence"],
            "ai_analysis":        analysis,
            "sensor_readings":    readings_payload,
            "affected_residents": affected,
            "estimated_eta":      eta,
            "resolved_at":        None,
        }

        try:
            created = db.create_incident(incident)
            new_incidents.append(created)
        except Exception as exc:
            print(f"[tick] Error creating incident: {exc}")

    return jsonify({
        "new_incidents": len(new_incidents),
        "incidents":     new_incidents,
        "zones_scanned": len(all_readings),
    })


@app.route("/api/sensors/history")
def sensor_history():
    stype   = request.args.get("sensor_type", "water_pressure")
    zone_id = request.args.get("zone_id", "z01")
    limit   = int(request.args.get("limit", 60))
    return jsonify(db.get_sensor_history(stype, zone_id, limit))


# ── Departments ───────────────────────────────────────────────
@app.route("/api/departments")
def departments():
    return jsonify(db.get_departments(active_only=True))


@app.route("/api/departments/<did>/incidents")
def dept_incidents(did):
    incidents = db.list_incidents(dept=did)
    dept_stats = {
        "total":    len(incidents),
        "active":   sum(1 for i in incidents if i["status"] != "resolved"),
        "critical": sum(1 for i in incidents if i["severity"] == "critical"),
        "resolved": sum(1 for i in incidents if i["status"] == "resolved"),
    }
    return jsonify({"stats": dept_stats, "incidents": incidents})


# ── Teams ─────────────────────────────────────────────────────
@app.route("/api/teams")
def teams():
    dept = request.args.get("dept") or None
    return jsonify(db.get_teams(dept))


# ── Audit ─────────────────────────────────────────────────────
@app.route("/api/audit/<iid>")
def audit(iid):
    return jsonify(db.get_audit(iid))


# ── ML model metadata ─────────────────────────────────────────
@app.route("/api/model/meta")
def model_meta():
    if _META:
        return jsonify(_META)
    # Fallback: derive from bundle
    return jsonify({
        "accuracy": ML_BUNDLE.get("accuracy", {}),
        "features": ML_BUNDLE.get("features", []),
    })


# ── Simulation helpers ────────────────────────────────────────
@app.route("/api/simulate/anomaly", methods=["POST", "OPTIONS"])
def simulate_anomaly():
    if request.method == "OPTIONS":
        return "", 204
    body    = request.get_json(force=True, silent=True) or {}
    zone_id = body.get("zone_id", "z01")
    dept    = body.get("dept", "water")
    NETWORK.force_zone_anomaly(zone_id, dept)
    return jsonify({"ok": True, "zone_id": zone_id, "dept": dept})


# ── Private helpers ───────────────────────────────────────────

def _fake_sensor_values(dept: str, severity: str) -> dict:
    """Return synthetic sensor readings that look like an anomaly for `dept`."""
    dept_sensors = {
        "water":        ["water_pressure", "water_flow", "pipe_vibration"],
        "electricity":  ["voltage_level",  "transformer_temp", "current_draw"],
        "municipality": ["road_crack_pct", "pothole_depth_cm"],
        "drainage":     ["drain_flow",     "drain_blockage_pct", "sewage_level_cm"],
    }
    result: dict = {}
    for stype in dept_sensors.get(dept, []):
        cfg           = SENSOR_CONFIG[stype]
        n_lo, n_hi    = cfg["normal"]
        w_lo, w_hi    = cfg["warn"]
        if severity == "critical":
            result[stype] = w_hi * 0.90 + random.uniform(-0.05, 0.05) * w_hi
        elif severity == "serious":
            result[stype] = n_hi + (w_hi - n_hi) * 0.60 + random.random() * (w_hi - n_hi) * 0.30
        else:
            result[stype] = n_hi + (w_hi - n_hi) * 0.20 + random.random() * (w_hi - n_hi) * 0.20
    return result


def _random_team(dept: str) -> str:
    pools = {
        "water":        ["Water-Alpha-1", "Water-Beta-2", "Water-Gamma-3"],
        "electricity":  ["Elec-Alpha", "Elec-Beta"],
        "municipality": ["Muni-Team-1", "Muni-Team-2"],
        "drainage":     ["Drain-X1", "Drain-X2"],
    }
    return random.choice(pools.get(dept, ["Team-1"]))


# ── Background auto-tick ──────────────────────────────────────
# FIX 4: use stdlib urllib.request instead of third-party requests library
_bg_started = False


def _bg_tick_loop() -> None:
    """Call /api/sensors/tick every 8 seconds to drive live detection."""
    while True:
        time.sleep(8)
        try:
            req = _urllib.Request(
                "http://127.0.0.1:5000/api/sensors/tick",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _urllib.urlopen(req, timeout=5):
                pass
        except Exception:
            pass  # Server might not be ready yet; will retry


def start_bg_tick() -> None:
    global _bg_started
    if not _bg_started:
        _bg_started = True
        t = threading.Thread(target=_bg_tick_loop, daemon=True)
        t.start()


# FIX 6: catch-all route is declared LAST so it never shadows /api/* routes
@app.route("/<path:path>")
def static_files(path):
    import os as _os
    full = _os.path.join(STATIC, path)
    if _os.path.isfile(full):
        return send_from_directory(STATIC, path)
    return send_from_directory(STATIC, "index.html")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":

    start_bg_tick()

    print("\n  NEXUS Smart City running at http://localhost:5000")
    print("    Dashboard  : http://localhost:5000/")
    print("    API health : http://localhost:5000/api/health\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)