# SmartFlow - Nagpur AI Traffic Management

SmartFlow is a Python traffic simulation and operations dashboard for 20 Nagpur intersections. This version upgrades the existing project in place; it keeps the Q-learning signal controller and simulator while making the map controls, weather, congestion prediction, accidents, pedestrians, emergency priority, analytics, and emissions work from shared backend state.

## Architecture

```text
Browser dashboard (Leaflet + OpenStreetMap)
    |-- Server-Sent Events: live intersection and analytics state
    |-- JSON API: weather, traffic, accidents, emergency, AI mode
    v
ThreadingHTTPServer (backend/server.py)
    |-- TrafficGenerator: vehicles, weather, pedestrians, incidents, emissions
    |-- TrafficManagementSystem: per-intersection Q-learning controllers
    |-- CongestionPredictor: Random Forest or dependency-free heuristic
    v
Kaggle CSV / synthetic training data + saved Q-table checkpoints
```

## Features

- Nagpur OpenStreetMap with 20 named intersections
- Animated vehicles that respond to signal phases and accidents
- Adaptive signals, live countdowns, pedestrian crossing phases
- Congestion prediction and predictive heatmap
- Ambulance routing with a low-congestion green wave
- Persistent backend accident and emergency simulation
- Clear, rain, fog, and storm effects with humidity and temperature
- AI vs fixed-time comparison and wait-time history
- GPS nearest-signal lookup
- Live fuel-idling and CO2 estimates
- Threaded Python API so SSE and dashboard controls work together
- SmartFlow Mobility OS hybrid UI with a compact proportional map
- Animated navigation, cards, counters, traffic twin, alerts, and transitions
- Explainable AI decisions and a live city mobility health score
- 5, 15, and 30 minute congestion forecast panel
- What-if traffic and weather scenario modeling
- Pause, replay, and scrub controls for recent traffic states
- Fastest, low-emission, and low-congestion route comparison
- Multi-signal green-wave corridor builder
- Prioritized alert drawer and operations decision log

## Run

```powershell
cd "smartflow_v2\smartflow"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\server.py
```

Open `http://localhost:8000`.

To run another SmartFlow instance on a different port:

```powershell
$env:SMARTFLOW_PORT="8011"
python backend\server.py
```

The dashboard can also run without installing ML dependencies. In that case SmartFlow automatically uses the built-in congestion heuristic.

## Kaggle Training

The trainer supports the Kaggle **Traffic Prediction Dataset** schema (`DateTime`, `Junction`, `Vehicles`) and Metro Interstate-style traffic CSV columns.

1. Install and authenticate the Kaggle CLI.
2. Download and extract the dataset:

```powershell
kaggle datasets download -d fedesoriano/traffic-prediction-dataset -p data --unzip
```

3. Train and save the model:

```powershell
python train_model.py --csv data\traffic.csv --output models\congestion.pkl
```

4. Start SmartFlow using that CSV for startup training:

```powershell
$env:SMARTFLOW_DATASET="data\traffic.csv"
python backend\server.py
```

You can also call `POST /api/train` with JSON `{"dataset_path":"data/traffic.csv"}`.

## API

- `GET /api/status` - all live intersection states
- `GET /api/analytics` - queue, wait, weather, fuel, CO2, pedestrians
- `GET /api/comparison` - AI vs traditional signal results
- `GET /api/predict/<id>` - congestion forecast
- `POST /api/weather` - `{"condition":"rain"}`
- `POST /api/emergency` - `{"intersection_id":"INT_01"}`
- `POST /api/accident` - `{"intersection_id":"INT_01","active":true}`
- `POST /api/add_traffic` - add a traffic spike
- `POST /api/ai_mode` - switch adaptive/fixed signal control
- `GET /stream` - live SSE feed

## Tests

```powershell
python -m unittest tests.test_all -v
```

The suite covers traffic generation, weather telemetry, accidents, operational metrics, Q-learning, and prediction behavior.
