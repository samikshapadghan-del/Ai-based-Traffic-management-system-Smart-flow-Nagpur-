"""
Congestion Predictor
Trains a RandomForest on synthetic history to predict queue length 5 ticks ahead.
Fully offline, sklearn only.
"""
try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    np = None
    RandomForestRegressor = StandardScaler = None
    mean_absolute_error = None
    SKLEARN_AVAILABLE = False

import csv
from datetime import datetime
import math
import pickle
import os


class CongestionPredictor:
    """
    Predicts total queue length at each intersection N ticks ahead.
    Features: current queue, wait, time of day, weather, phase encoding.
    """

    def __init__(self, lookahead: int = 5):
        self.lookahead = lookahead
        self.model = RandomForestRegressor(
            n_estimators=140, max_depth=10, random_state=42, n_jobs=-1
        ) if SKLEARN_AVAILABLE else None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.data_source = "heuristic fallback"
        self.feature_names = [
            "total_queue", "avg_wait", "time_of_day", "weather_multiplier",
            "phase_ns", "lane_N_queue", "lane_S_queue", "lane_E_queue", "lane_W_queue",
            "hour_sin", "hour_cos"  # cyclic time encoding
        ]

    def _extract_features(self, state: dict) -> list:
        tod = state.get("time_of_day", 12)
        return [
            state.get("total_queue", 0),
            state.get("avg_wait", 0),
            tod,
            state.get("weather_multiplier", 1.0),
            1.0 if "NS" in state.get("phase", "") else 0.0,
            state.get("lanes", {}).get("N", {}).get("queue", 0),
            state.get("lanes", {}).get("S", {}).get("queue", 0),
            state.get("lanes", {}).get("E", {}).get("queue", 0),
            state.get("lanes", {}).get("W", {}).get("queue", 0),
            math.sin(2 * math.pi * tod / 24),
            math.cos(2 * math.pi * tod / 24),
        ]

    def generate_training_data(self, n_steps: int = 2000):
        """Generate synthetic history for training."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from simulator.traffic_generator import TrafficGenerator

        gen = TrafficGenerator(num_intersections=4, seed=99)
        history = []  # list of (features_t, queue_t+lookahead) per intersection

        buffer = []
        print(f"  Generating {n_steps} ticks of training data...")
        for step in range(n_steps + self.lookahead):
            states = gen.tick()
            for iid, s in states.items():
                buffer.append((iid, self._extract_features(s), s["total_queue"]))

        # Pair features at t with target at t+lookahead
        # Buffer has 4 intersections interleaved; group by intersection
        by_int = {}
        for iid, feats, queue in buffer:
            if iid not in by_int:
                by_int[iid] = []
            by_int[iid].append((feats, queue))

        X, y = [], []
        for iid, records in by_int.items():
            for i in range(len(records) - self.lookahead):
                X.append(records[i][0])
                y.append(records[i + self.lookahead][1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, n_steps: int = 2000):
        if not SKLEARN_AVAILABLE:
            self.is_trained = True
            self.data_source = "heuristic fallback (install requirements.txt for RandomForest)"
            return {"mae": 0.0, "train_samples": 0, "backend": "heuristic",
                    "data_source": self.data_source}
        print("Training congestion predictor...")
        X, y = self.generate_training_data(n_steps)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        preds = self.model.predict(X_test_s)
        mae = mean_absolute_error(y_test, preds)
        self.is_trained = True
        self.data_source = "SmartFlow synthetic traffic"
        print(f"  Trained on {len(X_train)} samples | MAE={mae:.2f} vehicles")
        return {
            "mae": round(float(mae), 2),
            "train_samples": len(X_train),
            "backend": "RandomForest",
            "data_source": self.data_source,
        }

    @staticmethod
    def _first(row: dict, names: tuple, default=None):
        lower = {str(k).lower(): v for k, v in row.items()}
        for name in names:
            value = lower.get(name.lower())
            if value not in (None, ""):
                return value
        return default

    def train_csv(self, path: str):
        """Train from common Kaggle traffic CSV schemas.

        Supports the Traffic Prediction dataset (DateTime, Junction, Vehicles)
        and Metro Interstate Traffic Volume-style columns.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("numpy and scikit-learn are required for CSV training")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        grouped = {}
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw_volume = self._first(
                    row, ("Vehicles", "traffic_volume", "volume", "vehicle_count", "Total")
                )
                if raw_volume is None:
                    continue
                try:
                    volume = max(0.0, float(raw_volume))
                except (TypeError, ValueError):
                    continue

                raw_dt = self._first(row, ("DateTime", "date_time", "timestamp", "datetime"), "")
                hour = 12.0
                if raw_dt:
                    normalized = str(raw_dt).replace("Z", "+00:00")
                    for parser in (
                        lambda value: datetime.fromisoformat(value),
                        lambda value: datetime.strptime(value, "%d/%m/%Y %H:%M"),
                        lambda value: datetime.strptime(value, "%m/%d/%Y %H:%M"),
                    ):
                        try:
                            parsed = parser(normalized)
                            hour = parsed.hour + parsed.minute / 60
                            break
                        except ValueError:
                            continue

                junction = str(self._first(row, ("Junction", "junction", "intersection_id"), "all"))
                humidity = float(self._first(row, ("humidity",), 50) or 50)
                rain = float(self._first(row, ("rain_1h", "precipitation"), 0) or 0)
                weather_mult = max(0.4, min(1.0, 1.0 - rain * 0.025 - max(0, humidity - 80) * 0.004))
                quarter = volume / 4.0
                state = {
                    "total_queue": volume,
                    "avg_wait": volume * 2.8,
                    "time_of_day": hour,
                    "weather_multiplier": weather_mult,
                    "phase": "NS_GREEN" if int(hour * 2) % 2 == 0 else "EW_GREEN",
                    "lanes": {d: {"queue": quarter} for d in "NSEW"},
                }
                grouped.setdefault(junction, []).append((self._extract_features(state), volume))

        X, y = [], []
        for records in grouped.values():
            for index in range(len(records) - self.lookahead):
                X.append(records[index][0])
                y.append(records[index + self.lookahead][1])
        if len(X) < 50:
            raise ValueError("CSV did not contain enough recognized traffic rows")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        self.model.fit(self.scaler.fit_transform(X_train), y_train)
        self.is_trained = True
        self.data_source = os.path.basename(path)
        mae = None
        if len(X_test):
            mae = float(mean_absolute_error(y_test, self.model.predict(self.scaler.transform(X_test))))
        return {
            "mae": round(mae, 2) if mae is not None else None,
            "train_samples": len(X_train),
            "backend": "RandomForest",
            "data_source": self.data_source,
        }

    def predict(self, state: dict) -> dict:
        """Predict queue size N ticks ahead for one intersection."""
        if not self.is_trained:
            queue = state.get("total_queue", 0)
            level = "low" if queue < 10 else "medium" if queue < 25 else "high"
            return {"predicted_queue": queue, "congestion_level": level,
                    "confidence": "untrained"}
        if not SKLEARN_AVAILABLE:
            queue = float(state.get("total_queue", 0))
            wait = float(state.get("avg_wait", 0))
            weather = float(state.get("weather_multiplier", 1.0))
            pred = max(0.0, queue * (1.08 + (1.0 - weather) * 0.35) + wait / 45.0)
            level = "low" if pred < 10 else "medium" if pred < 25 else "high"
            return {
                "predicted_queue": round(pred, 1),
                "lookahead_ticks": self.lookahead,
                "congestion_level": level,
                "confidence": "heuristic",
            }
        feats = np.array([self._extract_features(state)], dtype=np.float32)
        feats_s = self.scaler.transform(feats)
        pred = float(self.model.predict(feats_s)[0])
        level = "low" if pred < 10 else "medium" if pred < 25 else "high"
        return {
            "predicted_queue": round(max(0, pred), 1),
            "lookahead_ticks": self.lookahead,
            "congestion_level": level
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "lookahead": self.lookahead, "trained": self.is_trained,
                         "data_source": self.data_source}, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.lookahead = data["lookahead"]
        self.is_trained = data["trained"]
        self.data_source = data.get("data_source", "saved model")
        return True


if __name__ == "__main__":
    pred = CongestionPredictor(lookahead=5)
    metrics = pred.train(n_steps=1000)
    print(f"Metrics: {metrics}")

    # Test prediction
    test_state = {
        "total_queue": 20, "avg_wait": 45.0, "time_of_day": 8.5,
        "weather_multiplier": 1.0, "phase": "NS_GREEN",
        "lanes": {"N": {"queue": 8}, "S": {"queue": 6}, "E": {"queue": 4}, "W": {"queue": 2}}
    }
    result = pred.predict(test_state)
    print(f"Prediction: {result}")
