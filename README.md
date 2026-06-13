# SmartFlow — AI Traffic Management System

## Quick Start (VS Code)

```bash
# 1. Open folder in VS Code
code smartflow/

# 2. Create virtual environment (VS Code will detect it)
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install numpy pandas scikit-learn

# 4. Run tests (VS Code Testing panel or terminal)
python tests/test_all.py

# 5. Start the server
python backend/server.py

# 6. Open dashboard
# http://localhost:8000
```

## Architecture

```
smartflow/
├── backend/
│   └── server.py          # HTTP server + SSE + all API endpoints
├── ml/
│   ├── signal_controller.py   # Q-learning RL agent per intersection
│   └── congestion_predictor.py # RandomForest 5-tick ahead prediction
├── simulator/
│   └── traffic_generator.py   # Synthetic traffic (rush hour, incidents, emergencies)
├── tests/
│   └── test_all.py            # Full test suite (25 tests)
└── README.md
```

## Components

### Synthetic Traffic Generator
- Rush hour curves (8am + 6pm peaks)
- Poisson vehicle arrivals per lane
- Random incidents (2% per tick) and emergencies (1% per tick)
- Weather multiplier effects
- Configurable tick speed (default: 30s per tick)

### RL Signal Controller (Q-Learning)
- State: queue bucket + wait time bucket + time of day + phase
- Actions: 7 phase durations (10, 20, 30, 45, 60, 90, 120 seconds)
- Reward: −(total_queue × 2 + avg_wait)
- Emergency override: always 10s phase to cycle quickly
- Incident override: 90s green for congested direction
- Epsilon-greedy with decay (0.30 → 0.05)

### Congestion Predictor
- RandomForestRegressor (100 trees)
- Predicts total queue 5 ticks ahead (~2.5 minutes)
- Features: queue, wait, time-of-day (sin/cos encoded), weather, lane breakdown
- Trains in background on startup (~30s)
- Output: predicted_queue + congestion_level (low/medium/high)

### API Server (pure Python stdlib)
- `GET  /`                    — Live dashboard (HTML)
- `GET  /api/status`          — All intersection states (JSON)
- `GET  /api/stats`           — RL agent stats
- `GET  /api/predict/<id>`    — Congestion prediction
- `GET  /api/history`         — Wait time history array
- `POST /api/train`           — Retrain predictor
- `POST /api/phase/<id>`      — Override phase duration
- `GET  /stream`              — SSE live stream (used by dashboard)

### Dashboard
- Built-in, served at `/`
- Live intersection cards (queue, wait, phase, emergency badge)
- Congestion prediction per intersection
- Wait time history chart
- System-wide metrics bar

## VS Code Tips
- Install **Python** + **Pylance** extensions
- Set interpreter to `.venv/python`
- Run/debug `backend/server.py` via Run menu (F5)
- Use **REST Client** extension to test API endpoints
- `tests/test_all.py` shows in the Testing panel automatically

## Training Notes
Predictor trains automatically on startup (~800 synthetic ticks, ~30s).
To retrain manually: `POST http://localhost:8000/api/train`

RL agents train online — improve continuously as simulator runs.
Save checkpoints: agents call `tms.save_all("checkpoints/")` on shutdown (Ctrl+C).
