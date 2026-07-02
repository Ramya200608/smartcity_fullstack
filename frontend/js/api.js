/* NEXUS Smart City — API Client + Shared Utilities */
const API = {
  BASE: window.location.port === '5000' ? '' : 'http://localhost:5000',

  async get(path) {
    const r = await fetch(this.BASE + path);
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  },

  async post(path, body = {}) {
    const r = await fetch(this.BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  },

  // Incident endpoints
  incidents:   (params = {}) => API.get('/api/incidents?' + new URLSearchParams(params)),
  incident:    (id)         => API.get(`/api/incidents/${id}`),
  resolve:     (id, actor='operator') => API.post(`/api/incidents/${id}/resolve`, { actor }),
  assign:      (id, team, actor='operator') => API.post(`/api/incidents/${id}/assign`, { team, actor }),
  progress:    (id)         => API.post(`/api/incidents/${id}/progress`),

  // Sensor endpoints
  sensors:     (zone_id)    => API.get('/api/sensors' + (zone_id ? `?zone_id=${zone_id}` : '')),
  tick:        ()           => API.post('/api/sensors/tick'),
  sensorHistory: (type, zone) => API.get(`/api/sensors/history?sensor_type=${type}&zone_id=${zone}`),

  // Meta
  stats:       ()           => API.get('/api/stats'),
  departments: ()           => API.get('/api/departments'),
  deptInc:     (did)        => API.get(`/api/departments/${did}/incidents`),
  teams:       (dept)       => API.get('/api/teams' + (dept ? `?dept=${dept}` : '')),
  audit:       (id)         => API.get(`/api/audit/${id}`),
  modelMeta:   ()           => API.get('/api/model/meta'),
  health:      ()           => API.get('/api/health'),

  // Simulate
  simulate:    (zone_id, dept) => API.post('/api/simulate/anomaly', { zone_id, dept }),
  seed:        ()           => API.post('/api/simulate/seed'),
};

const H = {
  timeAgo(ts) {
    if (!ts) return '—';
    const s = Math.floor(Date.now() / 1000 - ts);
    if (s < 60)    return `${s}s ago`;
    if (s < 3600)  return `${Math.floor(s/60)}m ago`;
    if (s < 86400) return `${Math.floor(s/3600)}h ago`;
    return `${Math.floor(s/86400)}d ago`;
  },

  severityColor(s) {
    return { critical: '#ff3d57', serious: '#ffb300', moderate: '#00e5ff',
             resolved: '#00ff87', detected: '#9d4edd', normal: '#8fa3bf' }[s] || '#8fa3bf';
  },

  deptColor(d) {
    return { water: '#38bdf8', electricity: '#ffb300', municipality: '#00ff87', drainage: '#9d4edd' }[d] || '#8fa3bf';
  },

  deptIcon(d) {
    return { water: '💧', electricity: '⚡', municipality: '🏗', drainage: '🌊' }[d] || '📡';
  },

  deptLabel(d) {
    return { water: 'Water Dept.', electricity: 'Electricity Dept.',
             municipality: 'Municipality', drainage: 'Drainage Dept.' }[d] || d;
  },

  statusLabel(s) {
    return { detected: 'AI Detected', assigned: 'Team Assigned',
             in_progress: 'In Progress', resolved: 'Resolved' }[s] || s;
  },

  statusColor(s) {
    return { detected: '#9d4edd', assigned: '#ffb300', in_progress: '#00e5ff', resolved: '#00ff87' }[s] || '#8fa3bf';
  },

  fmt(n) {
    if (n === undefined || n === null) return '—';
    if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
    return String(n);
  },

  chip(text, cls) {
    return `<span class="chip chip-${cls}">${text}</span>`;
  },

  sensorStatus(val, normal, warn) {
    if (val >= normal[0] && val <= normal[1]) return ['normal', '#00ff87'];
    if (val >= warn[0]   && val <= warn[1])   return ['warning', '#ffb300'];
    return ['critical', '#ff3d57'];
  },

  toast(msg, type = 'success') {
    const stack = document.getElementById('toast-stack');
    if (!stack) return;
    const el = document.createElement('div');
    el.className = `toast t-${type}`;
    const icons = { success: '✅', warn: '⚠️', error: '❌', info: '📡' };
    el.innerHTML = `<span>${icons[type]||'📡'}</span><span>${msg}</span>`;
    stack.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(8px)'; el.style.transition = '.3s'; setTimeout(() => el.remove(), 350); }, 3500);
  },
};

// Clock
function startClock(id = 'clock') {
  const el = document.getElementById(id);
  if (!el) return;
  const tick = () => { el.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false }) + ' IST'; };
  tick();
  setInterval(tick, 1000);
}

// Auto-tick API every 8 seconds to drive new incident detection
let _polling = false;
function startPolling(onNewIncident) {
  if (_polling) return;
  _polling = true;
  setInterval(async () => {
    try {
      const res = await API.tick();
      if (res.new_incidents > 0 && onNewIncident) onNewIncident(res.incidents);
    } catch (e) {}
  }, 8000);
}