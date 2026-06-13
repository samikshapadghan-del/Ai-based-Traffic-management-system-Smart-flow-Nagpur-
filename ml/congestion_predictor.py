"""
Congestion Predictor
Trains a RandomForest on synthetic history to predict queue length 5 ticks ahead.
Fully offline, sklearn only.
"""
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
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
        self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
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
        print(f"  Trained on {len(X_train)} samples | MAE={mae:.2f} vehicles")
        return {"mae": round(float(mae), 2), "train_samples": len(X_train)}

    def predict(self, state: dict) -> dict:
        """Predict queue size N ticks ahead for one intersection."""
        if not self.is_trained:
            return {"predicted_queue": state.get("total_queue", 0), "confidence": "untrained"}
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
                         "lookahead": self.lookahead, "trained": self.is_trained}, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.lookahead = data["lookahead"]
        self.is_trained = data["trained"]
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
