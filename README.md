🏙 NEXUS Smart City — AI-Powered Urban Intelligence Platform

A full-stack AI smart city management system with real-time IoT sensor monitoring, machine learning incident detection, GPS tracking, and a beautiful Classic dashboard UI.

📋 Table of Contents
About the Project
Features
Tech Stack
Project Structure
Hardware Components
Installation
How to Run
API Endpoints
ML Model Details
Department Pages
Screenshots
Team
🌟 About the Project

NEXUS Smart City is an AI-powered platform built for Ahmedabad Municipal Corporation to monitor city infrastructure in real time. It uses 144 virtual IoT sensors across 12 city zones to automatically detect incidents, classify severity using machine learning, and track them via GPS — all without human reporting.

Built as a college final year project demonstrating:

Real-time IoT data processing
Machine learning for anomaly detection
Full-stack web development
Smart city infrastructure management
✨ Features
Feature	Description
🤖 AI Detection	Auto-detects incidents from sensor data
📍 GPS Tracking	Real-time incident location on city map
📊 ML Models	97.6–100% accuracy classification
💧 Water Dept	Pressure, flow & leak monitoring
⚡ Electricity Dept	Grid voltage & transformer health
🏗 Municipality Dept	Road crack & pothole detection
🌊 Drainage Dept	Flood risk & sewage level monitoring
🔔 Live Alerts	Real-time incident notifications
📱 Classic UI	Elegant navy/gold dashboard design
🛠 Tech Stack
Backend
Python 3.10+
Flask          — REST API framework
Scikit-learn   — Machine learning models
SQLite         — Database
NumPy / Pandas — Data processing
Frontend
HTML5 / CSS3 / JavaScript
Playfair Display — Display font
Lora             — Body font
Courier Prime    — Mono/data font
ML Models
RandomForest      — Problem type classification (100% accuracy)
GradientBoosting  — Severity classification (97.6% accuracy)
RandomForest      — Department routing (100% accuracy)
📁 Project Structure
smartcity_fullstack/
│
├── backend/
│   ├── app.py           ← Flask REST API (18 endpoints)
│   ├── ml_model.py      ← ML training & prediction
│   ├── iot_sensors.py   ← 144 virtual IoT sensors
│   ├── ai_engine.py     ← AI analysis & ETA estimation
│   ├── database.py      ← SQLite database layer
│   └── seed_demo.py     ← Demo data seeder
│
├── frontend/
│   ├── index.html                     ← Main dashboard
│   ├── css/
│   │   └── shared.css                 ← Global styles
│   ├── js/
│   │   └── api.js                     ← API helper functions
│   └── pages/departments/
│       ├── water.html                 ← Water department
│       ├── electricity.html           ← Electricity department
│       ├── municipality.html          ← Municipality department
│       └── drainage.html              ← Drainage department
│
├── models/
│   ├── nexus_ml_bundle.pkl            ← Trained ML models
│   └── model_meta.json                ← Model metadata
│
├── data/
│   └── nexus.db                       ← SQLite database
│
├── start.sh                           ← Quick start script
└── README.md                          ← This file
🔧 Hardware Components
💧 Water Department — ₹95
Component	Model	Price
Rain Sensor	YL-83	₹25
Float Switch	FS-01	₹25
Water Flow Sensor	YF-S201	₹45
🌊 Drainage Department — ₹85
Component	Model	Price
IR Sensor	FC-51	₹25
Vibration Sensor	SW-420	₹25
Ultrasonic Sensor	HC-SR04	₹35
🖥 Controller — ₹415
Component	Model	Price
Microcontroller	ESP32	₹280
USB Cable	Micro USB	₹50
Breadboard	Full Size	₹35
Jumper Wires	40 Pack	₹50

Total Hardware Cost: ₹595 only 🎉

⚙️ Installation
Step 1 — Clone the Repository
bash
git clone https://github.com/yourusername/nexus-smart-city.git
cd nexus-smart-city
Step 2 — Install Python Dependencies
bash
pip install flask scikit-learn numpy pandas
Step 3 — Verify Installation
bash
python --version     # Should be 3.10+
flask --version      # Should show Flask version
🚀 How to Run
Start the Server
bash
# Go to backend folder
cd smartcity_fullstack/backend

# Run the Flask server
python app.py
Open in Browser
http://localhost:5000
All Department Pages
Dashboard    → http://localhost:5000
Water        → http://localhost:5000/pages/departments/water.html
Electricity  → http://localhost:5000/pages/departments/electricity.html
Municipality → http://localhost:5000/pages/departments/municipality.html
Drainage     → http://localhost:5000/pages/departments/drainage.html
Stop the Server
Press Ctrl + C
🔌 API Endpoints
Method	Endpoint	Description
GET	/api/health	Server health check
GET	/api/stats	Overall statistics
GET	/api/incidents	List all incidents
GET	/api/incidents/<id>	Single incident detail
POST	/api/incidents/<id>/resolve	Resolve an incident
POST	/api/incidents/<id>/assign	Assign team
POST	/api/incidents/<id>/progress	Update progress
GET	/api/sensors	All sensor readings
POST	/api/sensors/tick	Trigger sensor update
GET	/api/sensors/history	Sensor history
GET	/api/departments	Department list
GET	/api/departments/<id>/incidents	Dept incidents
GET	/api/teams	Team list
GET	/api/audit/<id>	Audit log
GET	/api/model/meta	ML model metadata
POST	/api/simulate/anomaly	Inject test anomaly
POST	/api/simulate/seed	Seed demo data
🤖 ML Model Details
Training Data
Samples    : 8,000 synthetic IoT sensor records
Features   : 12 sensor readings per zone
Train/Test : 80% / 20% split
Sensor Features Used
water_pressure      water_flow         pipe_vibration
voltage_level       transformer_temp   current_draw
road_crack_pct      pothole_depth_cm   drain_flow
drain_blockage_pct  sewage_level_cm    zone_rainfall_mm
Model Performance
Model	Algorithm	Accuracy
Problem Type	RandomForest (120 trees)	100%
Severity	GradientBoosting (100 estimators)	97.6%
Department	RandomForest (100 trees)	100%
Auto Detection Loop
Every 8 seconds:
  1. Tick all 144 IoT sensors
  2. Run ML prediction per zone
  3. Create incidents for new anomalies
  4. Send real-time alerts
🏛 Department Pages
💧 Water Department
Live pressure & flow gauges
Zone pressure health map
Pipe network schematic animation
AI leak detection recommendations
⚡ Electricity Department
Live voltage waveform graph
Distribution grid status map
Transformer health monitor
AI grid fault predictions
🏗 Municipality Department
Road condition map with pins
Road health index per street
Active work orders tracker
AI maintenance schedule
🌊 Drainage Department
Drainage network flow diagram
Flood risk percentage by zone
24-hour rainfall forecast chart
Live sewage level table
🎨 UI Theme — Classic
Base Color    → Warm Cream   #f8f5ee
Primary       → Deep Navy    #1a2744
Accent        → Gold         #c9a84c
Danger        → Burgundy     #8b1a2f
Success       → Forest Green #1e4d2b
Font Display  → Playfair Display
Font Body     → Lora
Font Mono     → Courier Prime
📊 System Architecture
┌─────────────────────────────────────────┐
│           IoT Sensor Network            │
│     144 Sensors × 12 City Zones        │
└─────────────────┬───────────────────────┘
                  │ Sensor Data
                  ▼
┌─────────────────────────────────────────┐
│           Flask REST API                │
│         (app.py — 18 endpoints)        │
└──────┬──────────┬──────────────┬────────┘
       │          │              │
       ▼          ▼              ▼
┌──────────┐ ┌────────┐ ┌──────────────┐
│ ML Model │ │SQLite  │ │  AI Engine   │
│ Predict  │ │  DB    │ │  Analysis    │
└──────────┘ └────────┘ └──────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Classic Dashboard UI            │
│   5 Pages × Department Monitoring      │
└─────────────────────────────────────────┘
👨‍💻 Circuit Connections (ESP32)
Water Sensors
YL-83 Rain    → GPIO 34 (AO), GPIO 35 (DO), 3.3V, GND
YF-S201 Flow  → GPIO 27, 5V, GND
Float Switch  → GPIO 26, GND
Drainage Sensors
FC-51 IR      → GPIO 25, 3.3V, GND
SW-420 Vib    → GPIO 33, 3.3V, GND
HC-SR04       → GPIO 32 (TRIG), GPIO 35 (ECHO), 5V, GND
🏆 Project Highlights
✅ Zero human reporting — AI detects everything automatically
✅ Under ₹600 total hardware cost
✅ 100% free software — Python, Flask, SQLite
✅ 97.6% ML accuracy on severity classification
✅ Real-time GPS incident tracking
✅ Classic UI — Elegant navy & gold design
✅ 4 departments monitored simultaneously
📞 Support

If you have any issues running the project:

Make sure Python 3.10+ is installed
Run pip install flask scikit-learn numpy pandas
Make sure you are in the backend/ folder
Run python app.py
