"""
Synthetic Traffic Generator
Simulates vehicle counts, queue lengths, and wait times for N intersections.
No real hardware or SUMO needed.
"""
import random
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class VehicleType(Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    BIKE = "bike"
    PEDESTRIAN = "pedestrian"


class Phase(Enum):
    NORTH_SOUTH_GREEN = "NS_GREEN"
    EAST_WEST_GREEN = "EW_GREEN"
    ALL_RED = "ALL_RED"
    PEDESTRIAN = "PED_GREEN"


@dataclass
class Lane:
    direction: str  # N, S, E, W
    queue: int = 0
    wait_time: float = 0.0
    vehicle_counts: Dict[str, int] = field(default_factory=lambda: {v.value: 0 for v in VehicleType})
    flow_rate: float = 0.0  # vehicles/min


@dataclass
class IntersectionState:
    id: str
    lanes: Dict[str, Lane]
    current_phase: Phase = Phase.NORTH_SOUTH_GREEN
    phase_elapsed: float = 0.0
    phase_duration: float = 30.0
    emergency_active: bool = False
    incident_active: bool = False
    accident_active: bool = False
    pedestrian_waiting: int = 0
    cycles_since_pedestrian: int = 0
    timestamp: float = field(default_factory=time.time)


class TrafficGenerator:
    """
    Synthetic traffic sim. Generates realistic patterns:
    - Morning/evening rush hour spikes
    - Random incidents
    - Emergency vehicle events
    - Weather effects (multiplier)
    """

    VEHICLE_WEIGHTS = [0.70, 0.08, 0.07, 0.10, 0.05]  # car bus truck bike ped

    # 20 real Nagpur intersections with GPS coordinates
    NAGPUR_INTERSECTIONS = [
        ("INT_01", "Sitabuldi",          21.1458, 79.0882),
        ("INT_02", "Empress Mall",       21.1412, 79.0927),
        ("INT_03", "Zero Mile",          21.1497, 79.0801),
        ("INT_04", "Ganeshpeth",         21.1369, 79.0789),
        ("INT_05", "Cotton Market",      21.1521, 79.0734),
        ("INT_06", "Dharampeth",         21.1331, 79.0650),
        ("INT_07", "Ramdaspeth",         21.1289, 79.0750),
        ("INT_08", "Sadar",              21.1388, 79.0984),
        ("INT_09", "Variety Square",     21.1447, 79.0756),
        ("INT_10", "Chhatrapati Sq",     21.1552, 79.0963),
        ("INT_11", "Laxmi Nagar",        21.1235, 79.1198),
        ("INT_12", "Manish Nagar",       21.1149, 79.0843),
        ("INT_13", "Hingna Road",        21.1082, 79.0421),
        ("INT_14", "Wardha Road",        21.1003, 79.0912),
        ("INT_15", "Ambazari",           21.1418, 79.0401),
        ("INT_16", "Bhandara Road",      21.1631, 79.1201),
        ("INT_17", "Kamptee Road",       21.1762, 79.1043),
        ("INT_18", "Jaripatka",          21.1589, 79.0621),
        ("INT_19", "Nandanvan",          21.1102, 79.1102),
        ("INT_20", "Trimurti Nagar",     21.1245, 79.0520),
    ]

    def __init__(self, num_intersections: int = 20, seed: int = 42):
        random.seed(seed)
        self.num_intersections = min(num_intersections, len(self.NAGPUR_INTERSECTIONS))
        self.intersections: Dict[str, IntersectionState] = {}
        self.intersection_meta: Dict[str, dict] = {}  # id -> {name, lat, lng}
        self.time_of_day: float = 8.0  # 24h float, starts at 8am
        self.weather_multiplier: float = 1.0
        self.weather_condition: str = "clear"  # clear/rain/fog/storm
        self.weather = {
            "temperature_c": 31.0,
            "humidity_pct": 42,
            "precipitation_mm": 0.0,
            "visibility_km": 10.0,
        }
        self.ai_mode: bool = True  # AI vs Traditional toggle
        self.emergency_vehicles: List[Dict] = []
        self._emergency_ticks: Dict[str, int] = {}
        self._accident_ticks: Dict[str, int] = {}
        self._tick = 0

        for i in range(self.num_intersections):
            iid, name, lat, lng = self.NAGPUR_INTERSECTIONS[i]
            self.intersections[iid] = IntersectionState(
                id=iid,
                lanes={d: Lane(direction=d) for d in ["N", "S", "E", "W"]}
            )
            self.intersection_meta[iid] = {"name": name, "lat": lat, "lng": lng}

    def _demand_multiplier(self) -> float:
        """Rush hour curve: peaks at 8am and 6pm."""
        h = self.time_of_day % 24
        morning = math.exp(-0.5 * ((h - 8.0) / 1.2) ** 2)
        evening = math.exp(-0.5 * ((h - 18.0) / 1.5) ** 2)
        base = 0.15
        return base + 0.85 * max(morning, evening)

    def _generate_arrivals(self, direction: str) -> int:
        """Poisson arrivals scaled by demand."""
        demand = self._demand_multiplier() * self.weather_multiplier
        # E-W slightly higher than N-S on average
        ew_bias = 1.3 if direction in ("E", "W") else 1.0
        lam = demand * ew_bias * 8  # avg 8 vehicles/tick at peak
        return max(0, int(random.gauss(lam, math.sqrt(lam) + 0.1)))

    def _update_queue(self, lane: Lane, phase: Phase, arrivals: int, blocked: bool = False):
        """Queue builds when phase is red, drains when green."""
        green_dirs = {"NS_GREEN": ("N", "S"), "EW_GREEN": ("E", "W")}
        active = green_dirs.get(phase.value, ())
        discharge = 0
        if lane.direction in active and not blocked:
            capacity = max(1, round(random.randint(4, 7) * self.weather_multiplier))
            discharge = min(lane.queue + arrivals, capacity)
        lane.queue = max(0, lane.queue + arrivals - discharge)
        lane.wait_time = lane.queue * random.uniform(2.5, 4.0)
        lane.flow_rate = discharge * 2  # vehicles/min approx

        # Random vehicle type breakdown
        if arrivals > 0:
            for _ in range(arrivals):
                vtype = random.choices(list(VehicleType), weights=self.VEHICLE_WEIGHTS)[0]
                lane.vehicle_counts[vtype.value] += 1

    def set_weather(self, condition: str):
        """Set weather: clear/rain/fog/storm"""
        profiles = {
            "clear": (1.0, 31.0, 42, 0.0, 10.0),
            "rain": (0.72, 25.0, 84, 5.5, 6.0),
            "fog": (0.58, 23.0, 91, 0.2, 1.8),
            "storm": (0.40, 22.0, 96, 18.0, 2.5),
        }
        condition = condition if condition in profiles else "clear"
        multiplier, temperature, humidity, precipitation, visibility = profiles[condition]
        self.weather_condition = condition
        self.weather_multiplier = multiplier
        self.weather = {
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "precipitation_mm": precipitation,
            "visibility_km": visibility,
        }

    def set_ai_mode(self, enabled: bool):
        self.ai_mode = enabled

    def add_traffic_spike(self, intersection_id: str, multiplier: float = 3.0):
        """Inject a traffic spike at a specific intersection."""
        if intersection_id in self.intersections:
            inter = self.intersections[intersection_id]
            for lane in inter.lanes.values():
                lane.queue = int(lane.queue + random.randint(10, 20) * multiplier)

    def trigger_emergency(self, intersection_id: str):
        """Force emergency at intersection."""
        if intersection_id in self.intersections:
            self.intersections[intersection_id].emergency_active = True
            self._emergency_ticks[intersection_id] = 6

    def set_accident(self, intersection_id: str, active: bool = True):
        """Create or clear a persistent accident in simulation state."""
        if intersection_id not in self.intersections:
            return False
        self.intersections[intersection_id].accident_active = active
        if active:
            self._accident_ticks[intersection_id] = 30
            self.add_traffic_spike(intersection_id, 2.5)
        else:
            self._accident_ticks.pop(intersection_id, None)
        return True

    def tick(self, delta_seconds: float = 30.0):
        """Advance simulation by delta_seconds."""
        self._tick += 1
        self.time_of_day = (self.time_of_day + delta_seconds / 3600) % 24

        # Random weather event (3% chance per tick) — only if not manually set
        # Manual weather remains stable until changed through the API.

        # Random incident (2% chance)
        incident_id = random.choice(list(self.intersections.keys()))
        incident_active = random.random() < 0.02

        # Random emergency vehicle (1% chance)
        emergency_active = random.random() < 0.01

        states = {}
        for iid, inter in self.intersections.items():
            inter.timestamp = time.time()
            inter.phase_elapsed += delta_seconds
            inter.incident_active = (iid == incident_id and incident_active)

            if emergency_active and iid == incident_id:
                self._emergency_ticks[iid] = 4
            if self._emergency_ticks.get(iid, 0) > 0:
                inter.emergency_active = True
                self._emergency_ticks[iid] -= 1
            else:
                inter.emergency_active = False
                self._emergency_ticks.pop(iid, None)

            if self._accident_ticks.get(iid, 0) > 0:
                inter.accident_active = True
                self._accident_ticks[iid] -= 1
            elif inter.accident_active:
                inter.accident_active = False
                self._accident_ticks.pop(iid, None)

            inter.pedestrian_waiting = min(
                80, inter.pedestrian_waiting + random.randint(0, 3)
            )

            # Advance phase if duration elapsed
            if inter.phase_elapsed >= inter.phase_duration:
                inter.phase_elapsed = 0.0
                if inter.current_phase == Phase.PEDESTRIAN:
                    inter.current_phase = Phase.NORTH_SOUTH_GREEN
                    inter.pedestrian_waiting = max(0, inter.pedestrian_waiting - random.randint(8, 18))
                    inter.cycles_since_pedestrian = 0
                elif inter.pedestrian_waiting >= 8 and inter.cycles_since_pedestrian >= 2:
                    inter.current_phase = Phase.PEDESTRIAN
                    inter.phase_duration = 15.0
                else:
                    phases = [Phase.NORTH_SOUTH_GREEN, Phase.EAST_WEST_GREEN]
                    inter.current_phase = phases[(phases.index(inter.current_phase) + 1) % 2] \
                        if inter.current_phase in phases else Phase.NORTH_SOUTH_GREEN
                    inter.cycles_since_pedestrian += 1

            for direction, lane in inter.lanes.items():
                arrivals = self._generate_arrivals(direction)
                # Incident doubles queue buildup at that intersection
                if inter.incident_active or inter.accident_active:
                    arrivals = int(arrivals * 2.5)
                self._update_queue(lane, inter.current_phase, arrivals, inter.accident_active)

            states[iid] = self._serialize(inter)

        return states

    def set_phase_duration(self, intersection_id: str, duration: float):
        """Signal controller calls this to update phase timing."""
        if intersection_id in self.intersections:
            self.intersections[intersection_id].phase_duration = max(10.0, min(120.0, duration))

    def _serialize(self, inter: IntersectionState) -> dict:
        meta = self.intersection_meta.get(inter.id, {})
        ai_wait   = round(sum(l.wait_time for l in inter.lanes.values()) / 4, 1)
        imbalance = abs(
            (inter.lanes["N"].queue + inter.lanes["S"].queue)
            - (inter.lanes["E"].queue + inter.lanes["W"].queue)
        )
        trad_wait = round(ai_wait * 1.22 + imbalance * 0.9 + 4.0, 1)
        total_queue = sum(l.queue for l in inter.lanes.values())
        fuel_litres = total_queue * 0.8 * ai_wait / 3600
        co2_kg = fuel_litres * 2.31
        return {
            "id": inter.id,
            "name": meta.get("name", inter.id),
            "lat": meta.get("lat", 21.1458),
            "lng": meta.get("lng", 79.0882),
            "timestamp": inter.timestamp,
            "time_of_day": round(self.time_of_day, 2),
            "phase": inter.current_phase.value,
            "phase_elapsed": round(inter.phase_elapsed, 1),
            "phase_duration": round(inter.phase_duration, 1),
            "countdown": max(0, round(inter.phase_duration - inter.phase_elapsed, 1)),
            "emergency": inter.emergency_active,
            "incident": inter.incident_active,
            "accident": inter.accident_active,
            "pedestrian_waiting": inter.pedestrian_waiting,
            "pedestrian_crossing": inter.current_phase == Phase.PEDESTRIAN,
            "weather_multiplier": round(self.weather_multiplier, 2),
            "weather_condition": self.weather_condition,
            "weather": self.weather.copy(),
            "ai_mode": self.ai_mode,
            "lanes": {
                d: {
                    "queue": lane.queue,
                    "wait_time": round(lane.wait_time, 1),
                    "flow_rate": round(lane.flow_rate, 1),
                    "vehicle_counts": lane.vehicle_counts.copy()
                }
                for d, lane in inter.lanes.items()
            },
            "total_queue": total_queue,
            "avg_wait": round(ai_wait, 1),
            "traditional_wait": round(trad_wait, 1),
            "ai_wait": round(ai_wait, 1),
            "fuel_litres": round(fuel_litres, 3),
            "co2_kg": round(co2_kg, 3),
        }


if __name__ == "__main__":
    gen = TrafficGenerator(num_intersections=4)
    print("Ticking 5 steps...")
    for i in range(5):
        states = gen.tick()
        for iid, s in states.items():
            print(f"  {iid} | phase={s['phase']:15s} | queue={s['total_queue']:3d} | wait={s['avg_wait']:5.1f}s | emergency={s['emergency']}")
        print()
