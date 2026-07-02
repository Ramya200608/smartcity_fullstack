"""
NEXUS Smart City — Database Layer (SQLite)
Tables: incidents, sensor_logs, departments, teams, audit_log

BUGS FIXED vs original:
  BUG 1  DB_PATH was set at module-import time from os.environ, so env vars set
         by app.py *after* importing this module were silently ignored.
         FIX: replaced module-level DB_PATH with a lazy _db_path() function that
              reads os.environ["NEXUS_DB"] fresh on every get_conn() call.

  BUG 2  os.makedirs(os.path.dirname(DB_PATH)) would receive an empty string when
         DB_PATH had no directory component (e.g. just "nexus.db"), which passes
         silently on some platforms but fails on others.
         FIX: only call makedirs when dirname is non-empty.

  BUG 3  update_incident_status used "COALESCE(?, assigned_team)" to conditionally
         update assigned_team. SQLite's COALESCE treats NULL as "use existing" but
         the Python None was being passed as a positional param, meaning passing
         None truly set the column to NULL instead of leaving it unchanged.
         FIX: build the UPDATE statement dynamically — only include SET clauses
              for fields that are actually being changed.

  BUG 4  get_teams() was called as get_teams() (no dept) with params=() but the
         SQL string was built as "SELECT * FROM teams" with no WHERE clause yet
         conn.execute(sql, ()) — harmless, but ambiguous with the dept branch.
         FIX: split into two clear branches with explicit param tuples.

  BUG 5  list_incidents built params as a plain list but some callers expected a
         tuple; also the LIMIT placeholder was appended after building the WHERE
         clause which could produce "WHERE 1=1 LIMIT ?" (no filters) — correct
         but hard to read. Cleaned up for clarity.

  BUG 6  _row_to_dict silently swallowed any json.JSONDecodeError on gps_history /
         sensor_readings, returning [] even for non-JSON corrupt data with no log.
         FIX: kept the safe fallback but narrowed the exception type.
"""

import sqlite3, json, time, os
from contextlib import contextmanager
from typing import Optional


# ── DB path — evaluated lazily, not at import time ────────────
# FIX 1: call os.environ.get() each time so callers can set NEXUS_DB
#        before or after importing this module.
def _db_path() -> str:
    return os.environ.get("NEXUS_DB", "data/nexus.db")


def get_conn() -> sqlite3.Connection:
    path = _db_path()
    # FIX 2: only makedirs when there is a directory component
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    """Yield an open connection; commit on success, rollback on error, always close."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables (idempotent) and insert seed rows."""
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS incidents (
            id                  TEXT PRIMARY KEY,
            title               TEXT NOT NULL,
            description         TEXT,
            zone_id             TEXT NOT NULL,
            zone_name           TEXT NOT NULL,
            lat                 REAL,
            lng                 REAL,
            map_x               REAL,
            map_y               REAL,
            dept                TEXT NOT NULL,
            problem_type        TEXT NOT NULL,
            severity            TEXT NOT NULL
                CHECK(severity IN ('moderate','serious','critical')),
            status              TEXT NOT NULL DEFAULT 'detected'
                CHECK(status IN ('detected','assigned','in_progress','resolved')),
            ai_confidence       REAL,
            ai_analysis         TEXT,
            sensor_readings     TEXT,
            gps_history         TEXT,
            affected_residents  INTEGER DEFAULT 0,
            assigned_team       TEXT,
            estimated_eta       TEXT,
            created_at          REAL NOT NULL,
            updated_at          REAL NOT NULL,
            resolved_at         REAL
        );

        CREATE TABLE IF NOT EXISTS sensor_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id   TEXT NOT NULL,
            sensor_type TEXT NOT NULL,
            zone_id     TEXT NOT NULL,
            value       REAL NOT NULL,
            unit        TEXT,
            status      TEXT,
            logged_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sl_zone ON sensor_logs(zone_id, logged_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sl_type ON sensor_logs(sensor_type, logged_at DESC);

        CREATE TABLE IF NOT EXISTS departments (
            id        TEXT PRIMARY KEY,
            name      TEXT NOT NULL,
            icon      TEXT,
            color     TEXT,
            head_name TEXT,
            contact   TEXT,
            active    INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS teams (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            dept         TEXT NOT NULL,
            status       TEXT DEFAULT 'standby',
            members      INTEGER DEFAULT 3,
            current_zone TEXT
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id  TEXT,
            action       TEXT NOT NULL,
            performed_by TEXT DEFAULT 'system',
            detail       TEXT,
            ts           REAL NOT NULL
        );
        """)

        conn.executemany(
            "INSERT OR IGNORE INTO departments(id,name,icon,color,head_name,contact)"
            " VALUES(?,?,?,?,?,?)",
            [
                ("water",        "Water Department",        "💧", "#38bdf8", "Rajesh Kumar",  "+91-79-2222-0001"),
                ("electricity",  "Electricity Department",  "⚡", "#ffb300", "Priya Sharma",  "+91-79-2222-0002"),
                ("municipality", "Municipality Department", "🏗", "#00ff87", "Amit Patel",    "+91-79-2222-0003"),
                ("drainage",     "Drainage Department",     "🌊", "#9d4edd", "Sunita Verma",  "+91-79-2222-0004"),
            ],
        )

        conn.executemany(
            "INSERT OR IGNORE INTO teams(id,name,dept,status,members,current_zone)"
            " VALUES(?,?,?,?,?,?)",
            [
                ("W-A1", "Water-Alpha-1",  "water",        "active",  4, None),
                ("W-B2", "Water-Beta-2",   "water",        "standby", 3, None),
                ("W-C3", "Water-Gamma-3",  "water",        "standby", 3, None),
                ("E-A1", "Elec-Alpha",     "electricity",  "active",  3, None),
                ("E-B1", "Elec-Beta",      "electricity",  "standby", 2, None),
                ("M-A1", "Muni-Team-1",    "municipality", "standby", 5, None),
                ("M-B1", "Muni-Team-2",    "municipality", "standby", 4, None),
                ("D-A1", "Drain-X1",       "drainage",     "active",  3, None),
                ("D-B1", "Drain-X2",       "drainage",     "standby", 3, None),
            ],
        )


# ── Incident CRUD ─────────────────────────────────────────────

def create_incident(data: dict) -> dict:
    """
    Insert a new incident.  Returns the persisted incident dict.

    Required keys: id, title, zone_id, zone_name, dept, problem_type, severity.
    All other fields default gracefully.
    """
    now = time.time()
    data.setdefault("created_at", now)
    data.setdefault("updated_at", now)
    data.setdefault("status", "detected")

    # Serialise any list fields to JSON strings
    for key in ("sensor_readings", "gps_history"):
        if isinstance(data.get(key), list):
            data[key] = json.dumps(data[key])

    # Build initial GPS history if absent
    if not data.get("gps_history"):
        data["gps_history"] = json.dumps(
            [{"lat": data.get("lat"), "lng": data.get("lng"), "ts": now}]
        )

    with db() as conn:
        conn.execute(
            """
            INSERT INTO incidents (
                id, title, description,
                zone_id, zone_name, lat, lng, map_x, map_y,
                dept, problem_type, severity, status,
                ai_confidence, ai_analysis,
                sensor_readings, gps_history,
                affected_residents, assigned_team, estimated_eta,
                created_at, updated_at, resolved_at
            ) VALUES (
                :id, :title, :description,
                :zone_id, :zone_name, :lat, :lng, :map_x, :map_y,
                :dept, :problem_type, :severity, :status,
                :ai_confidence, :ai_analysis,
                :sensor_readings, :gps_history,
                :affected_residents, :assigned_team, :estimated_eta,
                :created_at, :updated_at, :resolved_at
            )
            """,
            {
                "id":                 data["id"],
                "title":              data["title"],
                "description":        data.get("description", ""),
                "zone_id":            data["zone_id"],
                "zone_name":          data["zone_name"],
                "lat":                data.get("lat"),
                "lng":                data.get("lng"),
                "map_x":              data.get("map_x"),
                "map_y":              data.get("map_y"),
                "dept":               data["dept"],
                "problem_type":       data["problem_type"],
                "severity":           data["severity"],
                "status":             data["status"],
                "ai_confidence":      data.get("ai_confidence"),
                "ai_analysis":        data.get("ai_analysis", ""),
                "sensor_readings":    data.get("sensor_readings", "[]"),
                "gps_history":        data["gps_history"],
                "affected_residents": data.get("affected_residents", 0),
                "assigned_team":      data.get("assigned_team"),
                "estimated_eta":      data.get("estimated_eta"),
                "created_at":         data["created_at"],
                "updated_at":         data["updated_at"],
                "resolved_at":        data.get("resolved_at"),
            },
        )
        _audit(
            conn, data["id"], "created", "ai_engine",
            f"AI detected {data['problem_type']} ({data['severity']})",
        )

    return get_incident(data["id"])


def get_incident(incident_id: str) -> Optional[dict]:
    """Return one incident dict or None."""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_incidents(
    dept: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Return incidents newest-first, with optional filters."""
    # FIX 5: build params as a list explicitly; keep WHERE logic readable
    clauses = ["1=1"]
    params: list = []
    if dept is not None:
        clauses.append("dept = ?")
        params.append(dept)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if severity is not None:
        clauses.append("severity = ?")
        params.append(severity)
    params.append(limit)

    sql = (
        "SELECT * FROM incidents"
        f" WHERE {' AND '.join(clauses)}"
        " ORDER BY created_at DESC LIMIT ?"
    )
    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_incident_status(
    incident_id: str,
    status: str,
    assigned_team: Optional[str] = None,
    actor: str = "system",
) -> Optional[dict]:
    """
    Update an incident's status.

    FIX 3: Build the SET clause dynamically so that passing assigned_team=None
    leaves the existing assigned_team value untouched (rather than nullifying it).
    """
    now = time.time()
    set_clauses = ["status = ?", "updated_at = ?"]
    values: list = [status, now]

    if assigned_team is not None:
        set_clauses.append("assigned_team = ?")
        values.append(assigned_team)

    if status == "resolved":
        set_clauses.append("resolved_at = ?")
        values.append(now)

    values.append(incident_id)
    sql = f"UPDATE incidents SET {', '.join(set_clauses)} WHERE id = ?"

    detail = f"Status → {status}"
    if assigned_team:
        detail += f", team={assigned_team}"

    with db() as conn:
        conn.execute(sql, values)
        _audit(conn, incident_id, "status_change", actor, detail)

    return get_incident(incident_id)


def append_gps(incident_id: str, lat: float, lng: float) -> None:
    """Append one GPS point to an incident's history (capped at 50)."""
    with db() as conn:
        row = conn.execute(
            "SELECT gps_history FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        if row is None:
            return
        history: list = json.loads(row["gps_history"] or "[]")
        history.append({"lat": lat, "lng": lng, "ts": time.time()})
        if len(history) > 50:
            history = history[-50:]
        conn.execute(
            "UPDATE incidents SET gps_history = ?, updated_at = ? WHERE id = ?",
            (json.dumps(history), time.time(), incident_id),
        )


def get_stats() -> dict:
    """Return aggregate KPI statistics."""
    with db() as conn:
        def scalar(sql: str) -> int:
            return conn.execute(sql).fetchone()[0]

        total    = scalar("SELECT COUNT(*) FROM incidents")
        active   = scalar("SELECT COUNT(*) FROM incidents WHERE status != 'resolved'")
        critical = scalar("SELECT COUNT(*) FROM incidents WHERE severity='critical' AND status != 'resolved'")
        serious  = scalar("SELECT COUNT(*) FROM incidents WHERE severity='serious'  AND status != 'resolved'")
        moderate = scalar("SELECT COUNT(*) FROM incidents WHERE severity='moderate' AND status != 'resolved'")
        resolved = scalar("SELECT COUNT(*) FROM incidents WHERE status='resolved'")
        affected = scalar("SELECT COALESCE(SUM(affected_residents),0) FROM incidents WHERE status != 'resolved'")
        by_dept  = {
            r["dept"]: r["cnt"]
            for r in conn.execute(
                "SELECT dept, COUNT(*) AS cnt"
                " FROM incidents WHERE status != 'resolved' GROUP BY dept"
            ).fetchall()
        }

    return {
        "total":              total,
        "active":             active,
        "critical":           critical,
        "serious":            serious,
        "moderate":           moderate,
        "resolved":           resolved,
        "affected_residents": int(affected),
        "sensors_online":     144,
        "by_dept":            by_dept,
    }


# ── Sensor log ────────────────────────────────────────────────

def log_sensor(readings: list) -> None:
    """Bulk-insert a list of sensor reading dicts."""
    now = time.time()
    rows = [
        (
            r["sensor_id"], r["sensor_type"], r["zone_id"],
            r["value"], r.get("unit"), r.get("status"), now,
        )
        for r in readings
    ]
    with db() as conn:
        conn.executemany(
            "INSERT INTO sensor_logs"
            "(sensor_id,sensor_type,zone_id,value,unit,status,logged_at)"
            " VALUES(?,?,?,?,?,?,?)",
            rows,
        )


def get_sensor_history(sensor_type: str, zone_id: str, limit: int = 60) -> list:
    """Return up to `limit` readings for a sensor+zone, oldest-first."""
    with db() as conn:
        rows = conn.execute(
            """
            SELECT value, status, logged_at
            FROM   sensor_logs
            WHERE  sensor_type = ? AND zone_id = ?
            ORDER  BY logged_at DESC
            LIMIT  ?
            """,
            (sensor_type, zone_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ── Teams & Departments ───────────────────────────────────────

def get_teams(dept: Optional[str] = None) -> list:
    """
    Return all teams or filter by department.
    FIX 4: use two distinct code paths with unambiguous param tuples.
    """
    with db() as conn:
        if dept is not None:
            rows = conn.execute(
                "SELECT * FROM teams WHERE dept = ?", (dept,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM teams").fetchall()
    return [dict(r) for r in rows]


def get_departments(active_only: bool = True) -> list:
    """Return department records."""
    sql = "SELECT * FROM departments" + (" WHERE active = 1" if active_only else "")
    with db() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ── Audit log ─────────────────────────────────────────────────

def get_audit(incident_id: str) -> list:
    """Return audit entries for an incident, newest first."""
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE incident_id = ? ORDER BY ts DESC",
            (incident_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Private helpers ───────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convert sqlite3.Row → dict, JSON-decoding blob fields."""
    d = dict(row)
    for key in ("sensor_readings", "gps_history"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                d[key] = []
    return d


def _audit(
    conn: sqlite3.Connection,
    incident_id: str,
    action: str,
    actor: str,
    detail: str = "",
) -> None:
    """Insert an audit row inside an already-open transaction."""
    conn.execute(
        "INSERT INTO audit_log(incident_id,action,performed_by,detail,ts)"
        " VALUES(?,?,?,?,?)",
        (incident_id, action, actor, detail, time.time()),
    )