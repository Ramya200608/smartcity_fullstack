"""
NEXUS Smart City — Virtual IoT Sensor Engine
Simulates 144 IoT sensors across 12 city zones.
Each sensor ticks independently with probabilistic anomaly injection.
"""

import numpy as np
import time, random, math
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

# ── City Zones ────────────────────────────────────────────────
CITY_ZONES = [
    {"id": "z01", "name": "Gandhi Nagar",       "lat": 23.2160, "lng": 72.6380, "map_x": 38, "map_y": 32},
    {"id": "z02", "name": "Sector 12 East",      "lat": 23.2200, "lng": 72.6450, "map_x": 58, "map_y": 24},
    {"id": "z03", "name": "Navrangpura",         "lat": 23.2100, "lng": 72.5560, "map_x": 22, "map_y": 48},
    {"id": "z04", "name": "Satellite",           "lat": 23.0200, "lng": 72.5070, "map_x": 14, "map_y": 64},
    {"id": "z05", "name": "Bopal",               "lat": 23.0300, "lng": 72.4670, "map_x": 8,  "map_y": 74},
    {"id": "z06", "name": "Maninagar",           "lat": 22.9970, "lng": 72.6010, "map_x": 46, "map_y": 80},
    {"id": "z07", "name": "Vatva Industrial",    "lat": 22.9610, "lng": 72.6410, "map_x": 62, "map_y": 86},
    {"id": "z08", "name": "Chandkheda",          "lat": 23.1150, "lng": 72.5990, "map_x": 44, "map_y": 18},
    {"id": "z09", "name": "Gota",                "lat": 23.1110, "lng": 72.5770, "map_x": 32, "map_y": 20},
    {"id": "z10", "name": "Prahlad Nagar",       "lat": 23.0100, "lng": 72.5030, "map_x": 12, "map_y": 70},
    {"id": "z11", "name": "Thaltej",             "lat": 23.0460, "lng": 72.5010, "map_x": 10, "map_y": 58},
    {"id": "z12", "name": "Nikol",               "lat": 23.0400, "lng": 72.6520, "map_x": 70, "map_y": 60},
]

SENSOR_CONFIG = {
    "water_pressure":     {"unit": "PSI",   "normal": (40,80),   "warn": (20,100),  "dept": "water"},
    "water_flow":         {"unit": "L/min", "normal": (80,200),  "warn": (20,280),  "dept": "water"},
    "pipe_vibration":     {"unit": "mm/s",  "normal": (0,2),     "warn": (2,10),    "dept": "water"},
    "voltage_level":      {"unit": "V",     "normal": (215,245), "warn": (195,265), "dept": "electricity"},
    "transformer_temp":   {"unit": "°C",    "normal": (20,65),   "warn": (65,95),   "dept": "electricity"},
    "current_draw":       {"unit": "A",     "normal": (10,80),   "warn": (80,130),  "dept": "electricity"},
    "road_crack_pct":     {"unit": "%",     "normal": (0,10),    "warn": (10,40),   "dept": "municipality"},
    "pothole_depth_cm":   {"unit": "cm",    "normal": (0,2),     "warn": (2,12),    "dept": "municipality"},
    "drain_flow":         {"unit": "m³/h",  "normal": (50,200),  "warn": (200,320), "dept": "drainage"},
    "drain_blockage_pct": {"unit": "%",     "normal": (0,25),    "warn": (25,80),   "dept": "drainage"},
    "sewage_level_cm":    {"unit": "cm",    "normal": (0,40),    "warn": (40,90),   "dept": "drainage"},
    "zone_rainfall_mm":   {"unit": "mm/hr", "normal": (0,20),    "warn": (20,60),   "dept": "drainage"},
}


@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str
    zone_id: str
    zone_name: str
    value: float
    unit: str
    status: str          # "normal" | "warning" | "critical"
    timestamp: float
    anomaly_strength: float = 0.0


class VirtualSensor:
    """Simulates a single IoT sensor with realistic drift and anomaly injection."""

    def __init__(self, sensor_id: str, sensor_type: str, zone: dict):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.zone = zone
        cfg = SENSOR_CONFIG[sensor_type]
        self.unit = cfg["unit"]
        self.normal_lo, self.normal_hi = cfg["normal"]
        self.warn_lo, self.warn_hi = cfg["warn"]
        self.dept = cfg["dept"]

        # State
        self.value = self._rand_normal()
        self._drift_target = self.value
        self._in_anomaly = False
        self._anomaly_strength = 0.0
        self._anomaly_cooldown = 0
        self._rng = np.random.default_rng(
            seed=int(abs(hash(sensor_id)) % (2**31))
        )

    def _rand_normal(self):
        lo, hi = self.normal_lo, self.normal_hi
        return random.uniform(lo, hi)

    def _rand_anomaly(self):
        """Return a value in the warning/critical range."""
        strength = self._rng.uniform(0.3, 1.0)
        self._anomaly_strength = float(strength)
        # Randomly go high or low depending on sensor type
        if self._rng.random() < 0.5:
            return self.warn_lo + (self.normal_lo - self.warn_lo) * (1 - strength * 0.8)
        else:
            return self.normal_hi + (self.warn_hi - self.normal_hi) * strength

    def tick(self) -> SensorReading:
        """Advance simulation by one tick and return a reading."""
        # Anomaly injection: ~0.4% chance per tick to start anomaly
        if not self._in_anomaly and self._anomaly_cooldown <= 0:
            if self._rng.random() < 0.004:
                self._in_anomaly = True
                self._drift_target = self._rand_anomaly()
                self._anomaly_cooldown = int(self._rng.integers(30, 180))  # last 30-180 ticks

        if self._in_anomaly:
            self._anomaly_cooldown -= 1
            if self._anomaly_cooldown <= 0:
                self._in_anomaly = False
                self._anomaly_strength = 0.0
                self._drift_target = self._rand_normal()
                self._anomaly_cooldown = int(self._rng.integers(60, 300))

        # Drift toward target with noise
        self.value += (self._drift_target - self.value) * 0.10
        normal_range = self.normal_hi - self.normal_lo
        self.value += float(self._rng.normal(0, normal_range * 0.015))

        # Clamp
        abs_lo = self.warn_lo * 0.3
        abs_hi = self.warn_hi * 1.3
        self.value = float(np.clip(self.value, abs_lo, abs_hi))

        # Status
        if self.normal_lo <= self.value <= self.normal_hi:
            status = "normal"
        elif self.warn_lo <= self.value <= self.warn_hi:
            status = "warning"
        else:
            status = "critical"

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            zone_id=self.zone["id"],
            zone_name=self.zone["name"],
            value=round(self.value, 3),
            unit=self.unit,
            status=status,
            timestamp=time.time(),
            anomaly_strength=round(self._anomaly_strength, 3),
        )

    def force_anomaly(self, strength: float = 0.9):
        """Manually trigger an anomaly (for testing)."""
        self._in_anomaly = True
        self._anomaly_strength = strength
        self._drift_target = self._rand_anomaly()
        self._anomaly_cooldown = 60


class SensorNetwork:
    """
    Manages all 144 virtual sensors across 12 zones.
    One sensor per type per zone = 12 zones × 12 types = 144 sensors.
    """

    def __init__(self):
        self.sensors: dict[str, VirtualSensor] = {}
        self._init_sensors()

    def _init_sensors(self):
        for zone in CITY_ZONES:
            for stype in SENSOR_CONFIG:
                sid = f"{zone['id']}_{stype}"
                self.sensors[sid] = VirtualSensor(sid, stype, zone)

    def tick_all(self) -> dict[str, dict]:
        """
        Tick every sensor. Returns:
          { zone_id: { sensor_type: SensorReading } }
        """
        results = {}
        for sid, sensor in self.sensors.items():
            reading = sensor.tick()
            zid = reading.zone_id
            if zid not in results:
                results[zid] = {}
            results[zid][reading.sensor_type] = asdict(reading)
        return results

    def get_zone_readings(self, zone_id: str) -> dict:
        """Get latest readings for a specific zone."""
        results = {}
        for sid, sensor in self.sensors.items():
            if sensor.zone["id"] == zone_id:
                reading = sensor.tick()
                results[reading.sensor_type] = asdict(reading)
        return results

    def get_all_latest(self) -> dict[str, dict]:
        """Return latest (non-ticking) value snapshot for all sensors."""
        results = {}
        for sid, sensor in self.sensors.items():
            zid = sensor.zone["id"]
            if zid not in results:
                results[zid] = {}
            results[zid][sensor.sensor_type] = {
                "sensor_id": sensor.sensor_id,
                "sensor_type": sensor.sensor_type,
                "zone_id": zid,
                "zone_name": sensor.zone["name"],
                "value": round(sensor.value, 3),
                "unit": sensor.unit,
                "status": "normal" if sensor.normal_lo <= sensor.value <= sensor.normal_hi else
                           "warning" if sensor.warn_lo <= sensor.value <= sensor.warn_hi else "critical",
                "timestamp": time.time(),
            }
        return results

    def force_zone_anomaly(self, zone_id: str, dept: str):
        """Force anomaly in all sensors of a dept in a zone."""
        for sid, sensor in self.sensors.items():
            if sensor.zone["id"] == zone_id and sensor.dept == dept:
                sensor.force_anomaly()


# Singleton
_network: Optional[SensorNetwork] = None

def get_network() -> SensorNetwork:
    global _network
    if _network is None:
        _network = SensorNetwork()
    return _network
