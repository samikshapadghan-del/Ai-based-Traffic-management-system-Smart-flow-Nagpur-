"""
Adaptive Signal Controller
Uses a Q-table RL agent to optimize phase durations.
Trains online from simulator feedback — no GPU needed.
"""
import random
import math
import json
import os
from collections import defaultdict
from typing import Dict, Tuple


class SignalController:
    """
    Q-learning agent per intersection.
    State: (queue_bucket, time_of_day_bucket, phase)
    Action: phase duration in seconds (discrete: 10,20,30,45,60,90,120)
    Reward: negative total wait time
    """

    DURATIONS = [10, 20, 30, 45, 60, 90, 120]

    def __init__(self, intersection_id: str, alpha=0.1, gamma=0.95, epsilon=0.3):
        self.id = intersection_id
        self.alpha = alpha       # learning rate
        self.gamma = gamma       # discount
        self.epsilon = epsilon   # exploration
        self.q_table: Dict[Tuple, list] = defaultdict(lambda: [0.0] * len(self.DURATIONS))
        self.last_state = None
        self.last_action_idx = 3  # default 45s
        self.episode_rewards = []
        self.total_steps = 0
        self._emergency_override = False

    def _state_key(self, obs: dict) -> tuple:
        """Discretize observation into state bucket."""
        queue = obs.get("total_queue", 0)
        wait = obs.get("avg_wait", 0)
        tod = obs.get("time_of_day", 12)
        phase = obs.get("phase", "NS_GREEN")

        q_bucket = min(int(queue / 5), 10)       # 0-50+ → 0-10
        w_bucket = min(int(wait / 30), 5)         # 0-150+ → 0-5
        tod_bucket = int(tod / 3)                 # 0-7 (3h blocks)
        phase_bucket = 0 if "NS" in phase else 1

        return (q_bucket, w_bucket, tod_bucket, phase_bucket)

    def choose_action(self, obs: dict) -> float:
        """Epsilon-greedy action selection. Returns phase duration in seconds."""
        # Safety override: emergency → short phase (10s) to cycle quickly
        if obs.get("emergency", False):
            self._emergency_override = True
            return 10.0

        # Incident → longer green on congested direction
        if obs.get("incident", False):
            ns_queue = obs.get("lanes", {}).get("N", {}).get("queue", 0) + \
                       obs.get("lanes", {}).get("S", {}).get("queue", 0)
            ew_queue = obs.get("lanes", {}).get("E", {}).get("queue", 0) + \
                       obs.get("lanes", {}).get("W", {}).get("queue", 0)
            phase = obs.get("phase", "NS_GREEN")
            if "NS" in phase and ns_queue > ew_queue:
                return 90.0
            elif "EW" in phase and ew_queue > ns_queue:
                return 90.0

        state = self._state_key(obs)

        # Decay epsilon over time (less exploration as agent learns)
        self.epsilon = max(0.05, self.epsilon * 0.9999)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.DURATIONS) - 1)
        else:
            q_vals = self.q_table[state]
            action_idx = q_vals.index(max(q_vals))

        self.last_state = state
        self.last_action_idx = action_idx
        return float(self.DURATIONS[action_idx])

    def update(self, new_obs: dict):
        """Q-table update after observing new state. Call after each tick."""
        if self.last_state is None:
            return

        # Reward = negative total wait time (want to minimize)
        reward = -new_obs.get("total_queue", 0) * 2 - new_obs.get("avg_wait", 0)

        # Bonus for clearing emergency fast
        if self._emergency_override and not new_obs.get("emergency", False):
            reward += 100.0
            self._emergency_override = False

        new_state = self._state_key(new_obs)
        old_q = self.q_table[self.last_state][self.last_action_idx]
        max_future_q = max(self.q_table[new_state])

        # Bellman update
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[self.last_state][self.last_action_idx] = new_q

        self.episode_rewards.append(reward)
        self.total_steps += 1
        self.last_state = new_state

    def avg_reward(self, window: int = 100) -> float:
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-window:]
        return round(sum(recent) / len(recent), 2)

    def save(self, path: str):
        data = {
            "id": self.id,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "q_table": {str(k): v for k, v in self.q_table.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.total_steps = data.get("total_steps", 0)
        for k, v in data.get("q_table", {}).items():
            self.q_table[eval(k)] = v


class TrafficManagementSystem:
    """Manages all intersection controllers + simulator integration."""

    def __init__(self, intersection_ids: list):
        self.controllers = {iid: SignalController(iid) for iid in intersection_ids}
        self.history = []

    def step(self, states: dict, generator) -> dict:
        """One control loop iteration. Returns actions taken."""
        actions = {}
        for iid, obs in states.items():
            ctrl = self.controllers[iid]
            duration = ctrl.choose_action(obs)
            generator.set_phase_duration(iid, duration)
            actions[iid] = duration
        return actions

    def learn(self, new_states: dict):
        """Update all Q-tables after observing new states."""
        for iid, obs in new_states.items():
            self.controllers[iid].update(obs)

    def stats(self) -> dict:
        return {
            iid: {
                "epsilon": round(ctrl.epsilon, 3),
                "steps": ctrl.total_steps,
                "avg_reward_100": ctrl.avg_reward(100),
                "q_states_known": len(ctrl.q_table)
            }
            for iid, ctrl in self.controllers.items()
        }

    def save_all(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for iid, ctrl in self.controllers.items():
            ctrl.save(os.path.join(directory, f"{iid}.json"))

    def load_all(self, directory: str):
        for iid, ctrl in self.controllers.items():
            ctrl.load(os.path.join(directory, f"{iid}.json"))


if __name__ == "__main__":
    from simulator.traffic_generator import TrafficGenerator

    gen = TrafficGenerator(num_intersections=4)
    tms = TrafficManagementSystem(list(gen.intersections.keys()))

    print("Training 200 steps...")
    states = gen.tick()
    for step in range(200):
        actions = tms.step(states, gen)
        new_states = gen.tick()
        tms.learn(new_states)
        states = new_states

        if step % 50 == 0:
            total_wait = sum(s["avg_wait"] for s in states.values())
            print(f"  Step {step:3d} | total_wait={total_wait:.1f}s | " +
                  f"epsilon={list(tms.controllers.values())[0].epsilon:.3f}")

    print("\nFinal agent stats:")
    for iid, s in tms.stats().items():
        print(f"  {iid}: steps={s['steps']} | avg_reward={s['avg_reward_100']} | q_states={s['q_states_known']}")
