#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NEXUS Smart City — Full Stack Startup Script
#  Python 3 · Flask · SQLite · scikit-learn
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗"
echo "  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝"
echo "  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗"
echo "  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║"
echo "  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║"
echo "  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝"
echo "  Smart City AI Platform — Government Dashboard"
echo ""

# Install dependencies
echo "📦 Checking dependencies..."
pip install flask scikit-learn numpy pandas --quiet --break-system-packages 2>/dev/null || \
pip3 install flask scikit-learn numpy pandas --quiet 2>/dev/null || true

# Train/load ML models
echo "🤖 Loading ML models..."
python3 backend/ml_model.py 2>&1 | grep -E "Accuracy|✅|Training"

# Init database
echo "🗄  Initialising database..."
python3 -c "
import sys, os
sys.path.insert(0,'backend')
os.environ['NEXUS_DB']='data/nexus.db'
import database as db
db.init_db()
print('  Database ready')
"

echo ""
echo "🌐 Starting NEXUS Smart City Server..."
echo "   URL:  http://localhost:5000"
echo "   API:  http://localhost:5000/api/health"
echo ""
echo "   Pages:"
echo "   ├─ Government Dashboard : http://localhost:5000/"
echo "   ├─ Water Department     : http://localhost:5000/pages/departments/water.html"
echo "   ├─ Electricity Dept.    : http://localhost:5000/pages/departments/electricity.html"
echo "   ├─ Municipality         : http://localhost:5000/pages/departments/municipality.html"
echo "   └─ Drainage Dept.       : http://localhost:5000/pages/departments/drainage.html"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

export NEXUS_DB="$SCRIPT_DIR/data/nexus.db"
python3 -c "
import sys, os
sys.path.insert(0, 'backend')
os.environ['NEXUS_DB'] = os.environ.get('NEXUS_DB', 'data/nexus.db')
from app import app
app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
"