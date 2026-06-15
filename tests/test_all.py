"""
SmartFlow Test Suite
Run: python -m pytest tests/ -v  (or python tests/test_all.py directly)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.traffic_generator import TrafficGenerator, Phase
from ml.signal_controller import SignalController, TrafficManagementSystem
from ml.congestion_predictor import CongestionPredictor
from backend.server import DASHBOARD_HTML


class TestTrafficGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = TrafficGenerator(num_intersections=2, seed=1)

    def test_tick_returns_all_intersections(self):
        states = self.gen.tick()
        self.assertEqual(len(states), 2)

    def test_state_has_required_keys(self):
        states = self.gen.tick()
        for s in states.values():
            for key in ["id", "phase", "lanes", "total_queue", "avg_wait", "emergency", "incident"]:
                self.assertIn(key, s)

    def test_lanes_have_four_directions(self):
        states = self.gen.tick()
        for s in states.values():
            self.assertEqual(set(s["lanes"].keys()), {"N", "S", "E", "W"})

    def test_queue_non_negative(self):
        for _ in range(10):
            states = self.gen.tick()
            for s in states.values():
                self.assertGreaterEqual(s["total_queue"], 0)

    def test_set_phase_duration_clamped(self):
        iid = list(self.gen.intersections.keys())[0]
        self.gen.set_phase_duration(iid, 999)
        self.assertEqual(self.gen.intersections[iid].phase_duration, 120.0)
        self.gen.set_phase_duration(iid, 1)
        self.assertEqual(self.gen.intersections[iid].phase_duration, 10.0)

    def test_time_advances(self):
        t0 = self.gen.time_of_day
        self.gen.tick(delta_seconds=3600)
        self.assertAlmostEqual(self.gen.time_of_day, (t0 + 1.0) % 24, places=2)

    def test_emergency_flag_is_bool(self):
        states = self.gen.tick()
        for s in states.values():
            self.assertIsInstance(s["emergency"], bool)

    def test_weather_telemetry_changes(self):
        self.gen.set_weather("rain")
        state = next(iter(self.gen.tick().values()))
        self.assertEqual(state["weather_condition"], "rain")
        self.assertGreater(state["weather"]["humidity_pct"], 80)
        self.assertGreater(state["weather"]["precipitation_mm"], 0)

    def test_accident_persists_and_can_clear(self):
        iid = next(iter(self.gen.intersections))
        self.assertTrue(self.gen.set_accident(iid, True))
        self.assertTrue(self.gen.tick()[iid]["accident"])
        self.gen.set_accident(iid, False)
        self.assertFalse(self.gen.tick()[iid]["accident"])

    def test_operational_metrics_are_serialized(self):
        state = next(iter(self.gen.tick().values()))
        for key in ["countdown", "pedestrian_waiting", "fuel_litres", "co2_kg"]:
            self.assertIn(key, state)
        self.assertGreaterEqual(state["countdown"], 0)


class TestSignalController(unittest.TestCase):

    def setUp(self):
        self.ctrl = SignalController("INT_01")

    def _mock_obs(self, queue=10, wait=30, emergency=False, incident=False, tod=8.0):
        return {
            "total_queue": queue, "avg_wait": wait, "time_of_day": tod,
            "weather_multiplier": 1.0, "phase": "NS_GREEN",
            "emergency": emergency, "incident": incident,
            "lanes": {"N": {"queue": 3}, "S": {"queue": 3}, "E": {"queue": 2}, "W": {"queue": 2}}
        }

    def test_choose_action_returns_valid_duration(self):
        obs = self._mock_obs()
        d = self.ctrl.choose_action(obs)
        self.assertIn(d, [float(x) for x in SignalController.DURATIONS])

    def test_emergency_returns_short_phase(self):
        obs = self._mock_obs(emergency=True)
        d = self.ctrl.choose_action(obs)
        self.assertEqual(d, 10.0)

    def test_update_increases_step_count(self):
        obs = self._mock_obs()
        self.ctrl.choose_action(obs)
        self.ctrl.update(obs)
        self.assertEqual(self.ctrl.total_steps, 1)

    def test_epsilon_decays(self):
        obs = self._mock_obs()
        e0 = self.ctrl.epsilon
        for _ in range(100):
            self.ctrl.choose_action(obs)
        self.assertLess(self.ctrl.epsilon, e0)

    def test_epsilon_floor(self):
        self.ctrl.epsilon = 0.001
        obs = self._mock_obs()
        for _ in range(1000):
            self.ctrl.choose_action(obs)
        self.assertGreaterEqual(self.ctrl.epsilon, 0.05)


class TestTrafficManagementSystem(unittest.TestCase):

    def setUp(self):
        self.gen = TrafficGenerator(num_intersections=2, seed=7)
        self.tms = TrafficManagementSystem(list(self.gen.intersections.keys()))

    def test_step_returns_actions_for_all(self):
        states = self.gen.tick()
        actions = self.tms.step(states, self.gen)
        self.assertEqual(set(actions.keys()), set(self.gen.intersections.keys()))

    def test_stats_has_all_intersections(self):
        states = self.gen.tick()
        self.tms.step(states, self.gen)
        stats = self.tms.stats()
        for iid in self.gen.intersections:
            self.assertIn(iid, stats)

    def test_full_loop_100_steps(self):
        states = self.gen.tick()
        for _ in range(100):
            actions = self.tms.step(states, self.gen)
            new_states = self.gen.tick()
            self.tms.learn(new_states)
            states = new_states
        # Agent should have learned something
        for ctrl in self.tms.controllers.values():
            self.assertGreater(ctrl.total_steps, 0)
            self.assertGreater(len(ctrl.q_table), 0)


class TestCongestionPredictor(unittest.TestCase):

    def setUp(self):
        self.pred = CongestionPredictor(lookahead=3)

    def _mock_state(self):
        return {
            "total_queue": 15, "avg_wait": 40.0, "time_of_day": 9.0,
            "weather_multiplier": 1.0, "phase": "EW_GREEN",
            "lanes": {"N": {"queue": 4}, "S": {"queue": 4}, "E": {"queue": 4}, "W": {"queue": 3}}
        }

    def test_predict_untrained_returns_current_queue(self):
        s = self._mock_state()
        result = self.pred.predict(s)
        self.assertIn("predicted_queue", result)
        self.assertEqual(result["confidence"], "untrained")

    def test_train_and_predict(self):
        metrics = self.pred.train(n_steps=300)
        self.assertIn("mae", metrics)
        self.assertLess(metrics["mae"], 20)  # reasonable accuracy
        result = self.pred.predict(self._mock_state())
        self.assertIn("congestion_level", result)
        self.assertIn(result["congestion_level"], ["low", "medium", "high"])
        self.assertGreaterEqual(result["predicted_queue"], 0)

    def test_trained_flag_set_after_train(self):
        self.pred.train(n_steps=200)
        self.assertTrue(self.pred.is_trained)


class TestDashboard(unittest.TestCase):

    def test_hybrid_ui_feature_contract(self):
        required = [
            "buildHybridUI", "AI Signal Cockpit", "AI Forecast",
            "Operations Log", "Priority Routing", "whatif-result",
            "alert-drawer", "stateTimeline", "buildGreenWave", "city-score",
        ]
        for marker in required:
            self.assertIn(marker, DASHBOARD_HTML)

    def test_map_is_bounded_in_dashboard(self):
        self.assertIn(".map-panel{height:430px", DASHBOARD_HTML)
        self.assertIn("fullscreen-map", DASHBOARD_HTML)


if __name__ == "__main__":
    print("=" * 60)
    print("SmartFlow Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestTrafficGenerator, TestSignalController,
                TestTrafficManagementSystem, TestCongestionPredictor,
                TestDashboard]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
