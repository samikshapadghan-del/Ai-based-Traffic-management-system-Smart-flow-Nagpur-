"""
SmartFlow Backend Server
Pure Python stdlib HTTP server + WebSocket-style SSE for real-time dashboard.
No FastAPI/uvicorn needed.

Endpoints:
  GET  /api/status          - all intersection states
  GET  /api/stats           - RL agent stats
  GET  /api/predict/<id>    - congestion prediction for intersection
  POST /api/train           - trigger training run
  POST /api/phase/<id>      - manually set phase duration
  GET  /stream              - SSE stream (text/event-stream) for dashboard
"""
import json
import sys
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.traffic_generator import TrafficGenerator
from ml.signal_controller import TrafficManagementSystem
from ml.congestion_predictor import CongestionPredictor

# ─── Shared state ────────────────────────────────────────────────────────────
NUM_INTERSECTIONS = 20
TICK_INTERVAL = 2.0  # seconds between sim ticks

generator = TrafficGenerator(num_intersections=NUM_INTERSECTIONS)
intersection_ids = list(generator.intersections.keys())
tms = TrafficManagementSystem(intersection_ids)
predictor = CongestionPredictor(lookahead=5)

current_states: dict = {}
current_stats: dict = {}
sse_clients: list = []
sse_lock = threading.Lock()
state_lock = threading.Lock()

metrics = {
    "total_ticks": 0,
    "total_emergencies_handled": 0,
    "avg_wait_history": [],        # AI mode wait history
    "trad_wait_history": [],       # Traditional mode wait history
    "predictor_trained": False,
    "ai_mode": True,
    "weather_condition": "clear",
}

# ─── Simulation loop ──────────────────────────────────────────────────────────
def simulation_loop():
    global current_states, current_stats
    states = generator.tick()

    while True:
        actions = tms.step(states, generator)
        new_states = generator.tick()
        tms.learn(new_states)

        # Attach predictions
        for iid, s in new_states.items():
            if metrics["predictor_trained"]:
                s["prediction"] = predictor.predict(s)
            else:
                s["prediction"] = {"predicted_queue": None, "congestion_level": "unknown"}
            # Track emergency handling
            if s.get("emergency") and states.get(iid, {}).get("emergency") is False:
                metrics["total_emergencies_handled"] += 1

        metrics["total_ticks"] += 1
        avg_wait = sum(s["avg_wait"] for s in new_states.values()) / len(new_states)
        avg_trad = sum(s["traditional_wait"] for s in new_states.values()) / len(new_states)
        metrics["avg_wait_history"].append(round(avg_wait, 1))
        metrics["trad_wait_history"].append(round(avg_trad, 1))
        if len(metrics["avg_wait_history"]) > 300:
            metrics["avg_wait_history"].pop(0)
        if len(metrics["trad_wait_history"]) > 300:
            metrics["trad_wait_history"].pop(0)

        with state_lock:
            current_states = new_states
            current_stats = tms.stats()

        # Broadcast to SSE clients
        event_data = json.dumps({
            "states": new_states,
            "stats": current_stats,
            "metrics": {
                "total_ticks": metrics["total_ticks"],
                "avg_wait": round(avg_wait, 1),
                "avg_trad_wait": round(avg_trad, 1),
                "emergencies_handled": metrics["total_emergencies_handled"],
                "predictor_trained": metrics["predictor_trained"],
                "ai_mode": metrics["ai_mode"],
                "weather_condition": metrics["weather_condition"],
            }
        })
        with sse_lock:
            dead = []
            for client in sse_clients:
                try:
                    client["queue"].append(event_data)
                except Exception:
                    dead.append(client)
            for d in dead:
                sse_clients.remove(d)

        states = new_states
        time.sleep(TICK_INTERVAL)


def training_thread():
    """Train predictor in background."""
    print("[Backend] Training congestion predictor in background...")
    result = predictor.train(n_steps=800)
    metrics["predictor_trained"] = True
    print(f"[Backend] Predictor trained: {result}")


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class SmartFlowHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default access log

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._serve_dashboard()
        elif path == "/api/status":
            with state_lock:
                self._send_json({"intersections": current_states, "tick": metrics["total_ticks"]})
        elif path == "/api/stats":
            with state_lock:
                self._send_json({"agents": current_stats, "metrics": metrics})
        elif path.startswith("/api/predict/"):
            iid = path.split("/")[-1]
            with state_lock:
                s = current_states.get(iid)
            if not s:
                self._send_json({"error": "intersection not found"}, 404)
            else:
                self._send_json(predictor.predict(s))
        elif path == "/api/history":
            self._send_json({
                "avg_wait_history": metrics["avg_wait_history"],
                "trad_wait_history": metrics["trad_wait_history"],
            })
        elif path == "/api/intersections":
            # Returns all intersection metadata (id, name, lat, lng)
            meta = [
                {"id": iid, **generator.intersection_meta.get(iid, {})}
                for iid in generator.intersections.keys()
            ]
            self._send_json({"intersections": meta})
        elif path == "/api/comparison":
            # AI vs Traditional comparison stats
            ai_hist = metrics["avg_wait_history"]
            tr_hist = metrics["trad_wait_history"]
            n = min(len(ai_hist), len(tr_hist), 100)
            if n > 0:
                ai_avg = sum(ai_hist[-n:]) / n
                tr_avg = sum(tr_hist[-n:]) / n
                saving_pct = round((tr_avg - ai_avg) / tr_avg * 100, 1) if tr_avg > 0 else 0
            else:
                ai_avg = tr_avg = saving_pct = 0
            self._send_json({
                "ai_avg_wait": round(ai_avg, 1),
                "traditional_avg_wait": round(tr_avg, 1),
                "saving_percent": saving_pct,
                "ticks_compared": n,
            })
        elif path == "/stream":
            self._handle_sse()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if path == "/api/train":
            t = threading.Thread(target=training_thread, daemon=True)
            t.start()
            self._send_json({"status": "training started in background"})
        elif path.startswith("/api/phase/"):
            iid = path.split("/")[-1]
            duration = body.get("duration", 30)
            generator.set_phase_duration(iid, float(duration))
            self._send_json({"status": "ok", "intersection": iid, "duration": duration})
        elif path == "/api/weather":
            condition = body.get("condition", "clear")
            generator.set_weather(condition)
            metrics["weather_condition"] = condition
            self._send_json({"status": "ok", "weather": condition})
        elif path == "/api/emergency":
            iid = body.get("intersection_id")
            if iid and iid in generator.intersections:
                generator.trigger_emergency(iid)
                self._send_json({"status": "ok", "emergency_at": iid})
            else:
                self._send_json({"error": "invalid intersection_id"}, 400)
        elif path == "/api/add_traffic":
            iid = body.get("intersection_id")
            mult = float(body.get("multiplier", 3.0))
            if iid and iid in generator.intersections:
                generator.add_traffic_spike(iid, mult)
                self._send_json({"status": "ok", "spike_at": iid, "multiplier": mult})
            else:
                self._send_json({"error": "invalid intersection_id"}, 400)
        elif path == "/api/ai_mode":
            enabled = bool(body.get("enabled", True))
            generator.set_ai_mode(enabled)
            metrics["ai_mode"] = enabled
            self._send_json({"status": "ok", "ai_mode": enabled})
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_sse(self):
        """Server-Sent Events stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        import queue as Q
        q = Q.Queue(maxsize=10)
        client = {"queue": []}

        with sse_lock:
            sse_clients.append(client)

        try:
            while True:
                if client["queue"]:
                    data = client["queue"].pop(0)
                    msg = f"data: {data}\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                else:
                    # Heartbeat every 3s
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    time.sleep(3)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with sse_lock:
                if client in sse_clients:
                    sse_clients.remove(client)

    def _serve_dashboard(self):
        """Serve the built-in HTML dashboard."""
        html = DASHBOARD_HTML
        self._send_html(html)


# ─── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartFlow — Nagpur AI Traffic</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0f1117;--card:#1a1f2e;--border:#2d3748;--text:#e2e8f0;--sub:#94a3b8;--dim:#64748b;--blue:#60a5fa;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}
/* TOPBAR */
.topbar{background:var(--card);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;flex-shrink:0}
.logo{font-size:17px;font-weight:700;color:var(--blue);white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.conn{font-size:12px;color:var(--sub)}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tick-lbl{font-size:12px;color:var(--dim)}
/* CONTROLS BAR */
.controls{background:#141820;border-bottom:1px solid var(--border);padding:10px 20px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.ctrl-group{display:flex;align-items:center;gap:6px}
.ctrl-label{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.btn{padding:6px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;cursor:pointer;transition:all .15s;white-space:nowrap}
.btn:hover{background:#2d3748}
.btn.active{background:#1d4ed8;border-color:#3b82f6;color:#fff}
.btn.danger{background:#7f1d1d;border-color:#ef4444;color:#fca5a5}
.btn.warn{background:#713f12;border-color:#f59e0b;color:#fde68a}
.btn.success{background:#14532d;border-color:#22c55e;color:#86efac}
select{padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;cursor:pointer}
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:13px}
.toggle{position:relative;width:44px;height:24px}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;inset:0;background:#374151;border-radius:12px;cursor:pointer;transition:.3s}
.slider:before{content:"";position:absolute;height:18px;width:18px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.slider{background:#2563eb}
input:checked+.slider:before{transform:translateX(20px)}
/* METRICS */
.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:12px 20px;background:#141820}
@media(max-width:900px){.metrics{grid-template-columns:repeat(3,1fr)}}
.metric{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
.mlabel{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.mval{font-size:20px;font-weight:700;color:var(--blue)}
.msub{font-size:10px;color:#475569;margin-top:1px}
/* MAIN LAYOUT */
.main{display:flex;flex:1;overflow:hidden;min-height:0}
/* MAP */
.map-panel{flex:1.4;position:relative;min-width:0}
#map{width:100%;height:100%;min-height:400px}
.map-legend{position:absolute;bottom:16px;left:16px;z-index:1000;background:rgba(15,17,23,.92);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:11px;display:flex;flex-direction:column;gap:5px}
.leg-row{display:flex;align-items:center;gap:7px}
.leg-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.gps-btn{position:absolute;top:16px;right:16px;z-index:1000;background:var(--blue);color:#fff;border:none;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;font-weight:600}
/* SIDEBAR */
.sidebar{width:340px;flex-shrink:0;overflow-y:auto;display:flex;flex-direction:column;gap:0;border-left:1px solid var(--border)}
.sidebar-section{padding:14px 16px;border-bottom:1px solid var(--border)}
.sec-title{font-size:12px;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
/* CHART */
.chart-wrap{height:90px;position:relative}
.chart-wrap svg{width:100%;height:100%}
.chart-legend{display:flex;gap:14px;margin-top:6px}
.cleg{font-size:10px;display:flex;align-items:center;gap:4px;color:var(--sub)}
.cleg-line{width:20px;height:2px;border-radius:1px}
/* COMPARISON */
.cmp-bars{display:flex;flex-direction:column;gap:6px;margin-top:4px}
.cmp-row{display:flex;align-items:center;gap:8px;font-size:12px}
.cmp-lbl{width:80px;color:var(--sub);flex-shrink:0}
.cmp-bar-bg{flex:1;height:16px;background:#1e293b;border-radius:4px;overflow:hidden}
.cmp-bar-fill{height:100%;border-radius:4px;transition:width .6s}
.cmp-val{width:40px;text-align:right;font-weight:600;font-size:12px}
.saving-badge{margin-top:8px;text-align:center;font-size:13px;font-weight:700;color:#4ade80;background:#14532d;border-radius:8px;padding:6px}
/* INT LIST */
.int-list{display:flex;flex-direction:column;gap:6px;max-height:320px;overflow-y:auto}
.int-item{background:#0f1117;border-radius:8px;padding:8px 10px;cursor:pointer;border:1px solid transparent;transition:border .15s}
.int-item:hover,.int-item.selected{border-color:var(--blue)}
.int-header{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.int-name{font-size:13px;font-weight:600;flex:1}
.int-badges{display:flex;gap:4px}
.badge{font-size:9px;padding:2px 7px;border-radius:99px;font-weight:700}
.badge-red{background:#7f1d1d;color:#fca5a5}
.badge-yellow{background:#713f12;color:#fde68a}
.badge-blue{background:#1e3a5f;color:#93c5fd}
.int-lanes{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.int-lane{background:#1a1f2e;border-radius:4px;padding:3px 5px;text-align:center}
.int-lane-dir{font-size:9px;color:var(--dim)}
.int-lane-q{font-size:13px;font-weight:700}
.int-pred{font-size:10px;padding:2px 7px;border-radius:5px;display:inline-block;margin-top:5px}
.pred-low{background:#14532d;color:#4ade80}
.pred-medium{background:#713f12;color:#fde68a}
.pred-high{background:#7f1d1d;color:#fca5a5}
.pred-unknown{background:#1e293b;color:var(--dim)}
/* NEAREST */
.nearest-card{background:#0f1117;border-radius:8px;padding:10px;border:1px solid #2563eb}
.nearest-name{font-size:15px;font-weight:700;color:var(--blue);margin-bottom:6px}
.nearest-detail{font-size:12px;color:var(--sub);line-height:1.8}
/* WEATHER */
.weather-chips{display:flex;gap:6px;flex-wrap:wrap}
.wchip{padding:5px 12px;border-radius:99px;border:1px solid var(--border);font-size:12px;cursor:pointer;background:var(--card);transition:all .15s}
.wchip.active{background:#1d4ed8;border-color:#3b82f6;color:#fff}
/* FOOTER */
.footer{padding:8px 20px;text-align:center;font-size:10px;color:#374151;border-top:1px solid var(--border);flex-shrink:0}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="logo">⚡ SmartFlow</div>
  <div class="dot" id="dot"></div>
  <div class="conn" id="conn-status">Connecting...</div>
  <div class="topbar-right">
    <span class="tick-lbl" id="tick-display">tick 0</span>
    <span class="tick-lbl" id="weather-display">☀ clear</span>
  </div>
</div>

<!-- CONTROLS BAR -->
<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">AI Mode</span>
    <div class="toggle-wrap">
      <span style="font-size:12px;color:var(--dim)">Traditional</span>
      <label class="toggle">
        <input type="checkbox" id="ai-toggle" checked onchange="toggleAI(this.checked)">
        <span class="slider"></span>
      </label>
      <span style="font-size:12px;color:var(--blue)">AI</span>
    </div>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Intersection</span>
    <select id="int-select" onchange="updateActionTarget()">
      <option value="">— select —</option>
    </select>
  </div>
  <div class="ctrl-group">
    <button class="btn danger" onclick="triggerEmergency()">🚨 Emergency</button>
    <button class="btn warn" onclick="addTraffic()">🚗 Add Traffic</button>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Weather</span>
    <div class="weather-chips">
      <div class="wchip active" id="wchip-clear" onclick="setWeather('clear')">☀ Clear</div>
      <div class="wchip" id="wchip-rain" onclick="setWeather('rain')">🌧 Rain</div>
      <div class="wchip" id="wchip-fog" onclick="setWeather('fog')">🌫 Fog</div>
      <div class="wchip" id="wchip-storm" onclick="setWeather('storm')">⛈ Storm</div>
    </div>
  </div>
  <div class="ctrl-group" style="margin-left:auto">
    <button class="btn" onclick="locateUser()">📍 My Location</button>
  </div>
</div>

<!-- METRICS -->
<div class="metrics">
  <div class="metric"><div class="mlabel">AI Avg Wait</div><div class="mval" id="m-wait">—</div><div class="msub">seconds</div></div>
  <div class="metric"><div class="mlabel">Traditional</div><div class="mval" id="m-trad" style="color:#f59e0b">—</div><div class="msub">seconds (fixed)</div></div>
  <div class="metric"><div class="mlabel">Time Saved</div><div class="mval" id="m-save" style="color:#22c55e">—</div><div class="msub">vs traditional</div></div>
  <div class="metric"><div class="mlabel">Emergencies</div><div class="mval" id="m-emerg">0</div><div class="msub">handled</div></div>
  <div class="metric"><div class="mlabel">Total Ticks</div><div class="mval" id="m-ticks">0</div><div class="msub">steps</div></div>
  <div class="metric"><div class="mlabel">AI Predictor</div><div class="mval" id="m-ai" style="font-size:13px;padding-top:4px">Training...</div><div class="msub">status</div></div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- MAP -->
  <div class="map-panel">
    <div id="map"></div>
    <button class="gps-btn" onclick="locateUser()">📍 Locate Me</button>
    <div class="map-legend">
      <div style="font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:4px">Queue Level</div>
      <div class="leg-row"><div class="leg-dot" style="background:#22c55e"></div>Low (&lt;10)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#f59e0b"></div>Medium (10–25)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#ef4444"></div>High (&gt;25)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#a855f7"></div>Emergency</div>
      <div class="leg-row"><div class="leg-dot" style="background:#3b82f6;border:2px solid #fff"></div>Nearest</div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- AI vs Traditional Chart -->
    <div class="sidebar-section">
      <div class="sec-title">Wait Time: AI vs Traditional</div>
      <div class="chart-wrap">
        <svg id="cmp-chart" viewBox="0 0 300 80" preserveAspectRatio="none"></svg>
      </div>
      <div class="chart-legend">
        <div class="cleg"><div class="cleg-line" style="background:#3b82f6"></div>AI</div>
        <div class="cleg"><div class="cleg-line" style="background:#f59e0b"></div>Traditional</div>
      </div>
    </div>

    <!-- Live Comparison Bars -->
    <div class="sidebar-section">
      <div class="sec-title">AI vs Traditional (live)</div>
      <div class="cmp-bars">
        <div class="cmp-row">
          <div class="cmp-lbl">AI Signal</div>
          <div class="cmp-bar-bg"><div class="cmp-bar-fill" id="cmp-ai-bar" style="background:#3b82f6;width:0%"></div></div>
          <div class="cmp-val" id="cmp-ai-val" style="color:#60a5fa">—</div>
        </div>
        <div class="cmp-row">
          <div class="cmp-lbl">Traditional</div>
          <div class="cmp-bar-bg"><div class="cmp-bar-fill" id="cmp-tr-bar" style="background:#f59e0b;width:0%"></div></div>
          <div class="cmp-val" id="cmp-tr-val" style="color:#fbbf24">—</div>
        </div>
      </div>
      <div class="saving-badge" id="saving-badge">Calculating savings...</div>
    </div>

    <!-- Nearest Signal -->
    <div class="sidebar-section" id="nearest-section" style="display:none">
      <div class="sec-title">📍 Nearest Signal</div>
      <div class="nearest-card">
        <div class="nearest-name" id="nearest-name">—</div>
        <div class="nearest-detail" id="nearest-detail">—</div>
      </div>
    </div>

    <!-- Intersection List -->
    <div class="sidebar-section" style="flex:1">
      <div class="sec-title">All Intersections</div>
      <div class="int-list" id="int-list"></div>
    </div>

  </div>
</div>

<div class="footer">SmartFlow v2.0 — Nagpur AI Traffic Management — 20 Intersections Live</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let allStates = {};
let mapMarkers = {};
let userMarker = null;
let userLatLng = null;
let selectedInt = null;
const aiHistory = [], tradHistory = [];
const MAX_H = 80;

// ── Map init ───────────────────────────────────────────────────────────────
const map = L.map('map', {zoomControl: true}).setView([21.1458, 79.0882], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution:'&copy; OpenStreetMap',
  maxZoom:19
}).addTo(map);

function markerColor(s) {
  if (s.emergency) return '#a855f7';
  if (s.total_queue > 25) return '#ef4444';
  if (s.total_queue > 10) return '#f59e0b';
  return '#22c55e';
}

function makeIcon(color, isNearest) {
  const size = isNearest ? 22 : 16;
  const border = isNearest ? '#ffffff' : 'transparent';
  return L.divIcon({
    className:'',
    html:`<div style="width:${size}px;height:${size}px;border-radius:50%;background:${color};border:3px solid ${border};box-shadow:0 0 8px ${color}88"></div>`,
    iconSize:[size,size],
    iconAnchor:[size/2,size/2]
  });
}

function nearestId(lat, lng) {
  let best = null, bestDist = Infinity;
  for (const [id, s] of Object.entries(allStates)) {
    const d = Math.hypot(s.lat - lat, s.lng - lng);
    if (d < bestDist) { bestDist = d; best = id; }
  }
  return best;
}

function initMarkers(states) {
  for (const [id, s] of Object.entries(states)) {
    if (mapMarkers[id]) continue;
    const m = L.marker([s.lat, s.lng], {icon: makeIcon(markerColor(s), false)})
      .addTo(map)
      .bindPopup('');
    m.on('click', () => selectIntersection(id));
    mapMarkers[id] = m;
  }
}

function updateMarkers(states) {
  const nid = userLatLng ? nearestId(userLatLng[0], userLatLng[1]) : null;
  for (const [id, s] of Object.entries(states)) {
    const m = mapMarkers[id];
    if (!m) continue;
    const isNearest = id === nid;
    m.setIcon(makeIcon(markerColor(s), isNearest));
    m.setPopupContent(`
      <b>${s.name}</b><br>
      Queue: ${s.total_queue} | Wait: ${s.avg_wait}s<br>
      Phase: ${s.phase}<br>
      ${s.emergency ? '🚨 EMERGENCY' : ''} ${s.incident ? '⚠ INCIDENT' : ''}
    `);
  }
}

// ── GPS locate ─────────────────────────────────────────────────────────────
function locateUser() {
  if (!navigator.geolocation) { alert('GPS not available in browser'); return; }
  navigator.geolocation.getCurrentPosition(pos => {
    const {latitude: lat, longitude: lng} = pos.coords;
    userLatLng = [lat, lng];
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.marker([lat, lng], {
      icon: L.divIcon({
        className:'',
        html:'<div style="width:18px;height:18px;border-radius:50%;background:#3b82f6;border:3px solid #fff;box-shadow:0 0 12px #3b82f688"></div>',
        iconSize:[18,18], iconAnchor:[9,9]
      })
    }).addTo(map).bindPopup('📍 You are here').openPopup();
    map.setView([lat, lng], 15);
    showNearest(lat, lng);
  }, () => {
    // Fallback: use Nagpur center as demo
    userLatLng = [21.1458, 79.0882];
    showNearest(21.1458, 79.0882);
  });
}

function showNearest(lat, lng) {
  const nid = nearestId(lat, lng);
  if (!nid || !allStates[nid]) return;
  const s = allStates[nid];
  document.getElementById('nearest-section').style.display = '';
  document.getElementById('nearest-name').textContent = `${s.name} (${nid})`;
  document.getElementById('nearest-detail').innerHTML =
    `Queue: <b>${s.total_queue}</b> vehicles &nbsp;|&nbsp; Wait: <b>${s.avg_wait}s</b><br>
     Phase: <b>${s.phase}</b> &nbsp;|&nbsp; Duration: <b>${s.phase_duration}s</b><br>
     Congestion: <b>${(s.prediction||{}).congestion_level||'unknown'}</b>`;
  // Highlight on map
  updateMarkers(allStates);
}

// ── Controls ───────────────────────────────────────────────────────────────
function toggleAI(on) {
  fetch('/api/ai_mode', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled: on})});
}

function updateActionTarget() {
  selectedInt = document.getElementById('int-select').value || null;
}

function triggerEmergency() {
  const iid = selectedInt;
  if (!iid) { alert('Select an intersection first'); return; }
  fetch('/api/emergency', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid})});
}

function addTraffic() {
  const iid = selectedInt;
  if (!iid) { alert('Select an intersection first'); return; }
  fetch('/api/add_traffic', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({intersection_id: iid, multiplier: 3.0})});
}

function setWeather(cond) {
  fetch('/api/weather', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({condition: cond})});
  ['clear','rain','fog','storm'].forEach(c => {
    document.getElementById('wchip-'+c).classList.toggle('active', c === cond);
  });
  const icons = {clear:'☀',rain:'🌧',fog:'🌫',storm:'⛈'};
  document.getElementById('weather-display').textContent = (icons[cond]||'') + ' ' + cond;
}

// ── Intersection select ────────────────────────────────────────────────────
function selectIntersection(id) {
  selectedInt = id;
  document.getElementById('int-select').value = id;
  document.querySelectorAll('.int-item').forEach(el =>
    el.classList.toggle('selected', el.dataset.id === id));
  // Pan map
  const s = allStates[id];
  if (s) map.setView([s.lat, s.lng], 15);
}

// ── Charts ─────────────────────────────────────────────────────────────────
function drawCmpChart() {
  const svg = document.getElementById('cmp-chart');
  if (aiHistory.length < 2) return;
  const w = 300, h = 80;
  const all = [...aiHistory, ...tradHistory];
  const maxV = Math.max(...all, 1);
  function line(arr, color) {
    const pts = arr.map((v,i) => {
      const x = (i/(arr.length-1))*w;
      const y = h - (v/maxV)*(h-8) - 4;
      return `${x},${y}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" opacity=".9"/>`;
  }
  svg.innerHTML = line(tradHistory, '#f59e0b') + line(aiHistory, '#3b82f6');
}

// ── Sidebar intersection list ──────────────────────────────────────────────
function renderIntList(states) {
  const list = document.getElementById('int-list');
  list.innerHTML = Object.entries(states).map(([id, s]) => {
    const pred = s.prediction || {};
    const plevel = pred.congestion_level || 'unknown';
    const lanesHtml = ['N','S','E','W'].map(d => {
      const q = (s.lanes[d]||{}).queue||0;
      const col = q>25?'#ef4444':q>10?'#f59e0b':'#22c55e';
      return `<div class="int-lane"><div class="int-lane-dir">${d}</div><div class="int-lane-q" style="color:${col}">${q}</div></div>`;
    }).join('');
    return `<div class="int-item${selectedInt===id?' selected':''}" data-id="${id}" onclick="selectIntersection('${id}')">
      <div class="int-header">
        <div class="int-name">${s.name}</div>
        <div class="int-badges">
          ${s.emergency?'<span class="badge badge-red">🚨</span>':''}
          ${s.incident?'<span class="badge badge-yellow">⚠</span>':''}
          <span class="badge badge-blue">${s.phase.replace('_GREEN','')}</span>
        </div>
      </div>
      <div class="int-lanes">${lanesHtml}</div>
      <div class="int-pred pred-${plevel}">⟳ ${pred.predicted_queue??'?'} in 5 ticks — ${plevel}</div>
    </div>`;
  }).join('');
}

// ── Populate dropdown ──────────────────────────────────────────────────────
function populateSelect(states) {
  const sel = document.getElementById('int-select');
  if (sel.options.length > 1) return;
  Object.entries(states).forEach(([id, s]) => {
    const o = document.createElement('option');
    o.value = id; o.textContent = `${id} — ${s.name}`;
    sel.appendChild(o);
  });
}

// ── SSE ────────────────────────────────────────────────────────────────────
const es = new EventSource('/stream');
es.onopen = () => {
  document.getElementById('conn-status').textContent = 'Live';
  document.getElementById('dot').style.background = '#22c55e';
};
es.onerror = () => {
  document.getElementById('conn-status').textContent = 'Reconnecting...';
  document.getElementById('dot').style.background = '#ef4444';
};
es.onmessage = (e) => {
  const data = JSON.parse(e.data);
  const states = data.states || {};
  const m = data.metrics || {};

  allStates = states;

  // Metrics bar
  document.getElementById('m-wait').textContent = m.avg_wait + 's';
  document.getElementById('m-trad').textContent = m.avg_trad_wait + 's';
  const saved = m.avg_trad_wait > 0 ? Math.round((m.avg_trad_wait - m.avg_wait) / m.avg_trad_wait * 100) : 0;
  document.getElementById('m-save').textContent = saved + '%';
  document.getElementById('m-emerg').textContent = m.emergencies_handled;
  document.getElementById('m-ticks').textContent = m.total_ticks;
  document.getElementById('m-ai').textContent = m.predictor_trained ? '✓ Active' : 'Training...';
  document.getElementById('tick-display').textContent = 'tick ' + m.total_ticks;

  // Comparison bars
  const maxWait = Math.max(m.avg_wait, m.avg_trad_wait, 1);
  document.getElementById('cmp-ai-bar').style.width = (m.avg_wait/maxWait*100)+'%';
  document.getElementById('cmp-tr-bar').style.width = (m.avg_trad_wait/maxWait*100)+'%';
  document.getElementById('cmp-ai-val').textContent = m.avg_wait+'s';
  document.getElementById('cmp-tr-val').textContent = m.avg_trad_wait+'s';
  document.getElementById('saving-badge').textContent =
    saved > 0 ? `✓ AI saves ${saved}% wait time` : 'Collecting comparison data...';

  // History charts
  aiHistory.push(m.avg_wait);
  tradHistory.push(m.avg_trad_wait);
  if (aiHistory.length > MAX_H) aiHistory.shift();
  if (tradHistory.length > MAX_H) tradHistory.shift();
  drawCmpChart();

  // Map
  if (Object.keys(mapMarkers).length === 0) initMarkers(states);
  updateMarkers(states);

  // Sidebar
  populateSelect(states);
  renderIntList(states);

  // Nearest (if GPS active)
  if (userLatLng) showNearest(userLatLng[0], userLatLng[1]);

  // AI mode toggle sync
  document.getElementById('ai-toggle').checked = m.ai_mode !== false;
};
</script>
</body>
</html>"""


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    PORT = 8000

    # Start background training
    t_train = threading.Thread(target=training_thread, daemon=True)
    t_train.start()

    # Start simulation loop
    t_sim = threading.Thread(target=simulation_loop, daemon=True)
    t_sim.start()

    print(f"""
╔══════════════════════════════════════════╗
║   SmartFlow — AI Traffic Management      ║
╠══════════════════════════════════════════╣
║   Dashboard: http://localhost:{PORT}        ║
║   API:       http://localhost:{PORT}/api    ║
╚══════════════════════════════════════════╝

Endpoints:
  GET  /api/status          - all intersection states
  GET  /api/stats           - RL agent stats
  GET  /api/predict/<id>    - congestion prediction
  POST /api/train           - retrain predictor
  POST /api/phase/<id>      - set phase duration (body: {{"duration": 45}})
  GET  /stream              - SSE live stream

Press Ctrl+C to stop.
""")

    server = HTTPServer(("0.0.0.0", PORT), SmartFlowHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SmartFlow] Shutting down. Saving model checkpoints...")
        tms.save_all("checkpoints/")
        print("[SmartFlow] Done.")


if __name__ == "__main__":
    main()
