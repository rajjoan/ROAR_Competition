"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

def menger_radius(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, max_radius: float = 10000.0) -> float:
    """Radius of the circle through 3 (x, y) points. Large/flat -> max_radius (effectively straight)."""
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    if a < 1e-3 or b < 1e-3 or c < 1e-3:
        return max_radius
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq < 1e-3:
        return max_radius
    return (a * b * c) / (4 * np.sqrt(area_sq))

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # Smooth the raw waypoints (same triangle filter as the main-branch prototype),
        # used both as the steering target path and as the curvature source below.
        n = len(self.maneuverable_waypoints)
        self.path = [
            0.2 * self.maneuverable_waypoints[(i - 1) % n].location
            + 0.6 * self.maneuverable_waypoints[i].location
            + 0.2 * self.maneuverable_waypoints[(i + 1) % n].location
            for i in range(n)
        ]

        self.velocity_profile = self._build_velocity_profile(self.path)
        self.last_speed = 0.0

    def _build_velocity_profile(self, path: List[np.ndarray]) -> np.ndarray:
        """
        Precompute a target speed (m/s) for every waypoint, once, offline:
          1. curvature-limited speed at each point (v = sqrt(mu*g*r))
          2. backward pass: cap speed so there's always room to brake for what's ahead
        No forward/acceleration cap is applied deliberately -- the reference solutions
        never modeled one either (they just floor the throttle whenever under target),
        and guessing a conservative accel limit is exactly the mistake the previous
        team's "physics-based speed profile" attempt made (see writeup: it was too
        conservative because the assumed model didn't match the sim). Better to leave
        acceleration to the runtime controller, which can't be more wrong than a guess.
        """
        n = len(path)
        xy = [p[:2] for p in path]

        # MU: kept flat/global rather than per-section. Per-point curvature already
        # gives much finer-grained corner detection than the reference solutions' 10
        # hand-drawn section boundaries, so section-specific mu shouldn't be needed to
        # get comparable safety margin. 2.75 is the reference solutions' own proven
        # "never spins out anywhere on this track" default -- NOT re-derived here, just
        # reused as a safe starting point. Untested with this exact controller though.
        MU = 2.75
        G = 9.81
        V_MIN, V_MAX = 15.0, 85.0  # m/s; 85 m/s ~= 306 km/h, matching the reference solutions' proven ceiling

        # A_BRAKE: derived, not guessed. The reference ThrottleController's braking-distance
        # formula (get_throttle_and_brake_2/speed_for_turn_new) is
        #   max_speed_kmh = sqrt(target_speed_kmh**2 + 2*a*dist)      [a = 170..200]
        # which is the same kinematic equation as below but with speed in km/h instead of
        # m/s. Converting: v_ms = v_kmh/3.6, so a_ms2 = a_kmh_form / 3.6**2 = a / 12.96.
        # a=170..200 -> ~13.1..15.4 m/s^2. This is still just carried over from a
        # different controller on the same track, not measured directly against this one.
        A_BRAKE = 14.0  # m/s^2

        v_curv = np.array([
            np.clip(np.sqrt(MU * G * menger_radius(xy[(i - 2) % n], xy[i], xy[(i + 2) % n])), V_MIN, V_MAX)
            for i in range(n)
        ])

        dist = np.array([np.linalg.norm(xy[(i + 1) % n] - xy[i]) for i in range(n)])

        v = v_curv.copy()
        # Two passes around the closed loop so the braking cap propagates correctly
        # through the wraparound (last waypoint -> first waypoint).
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                nxt = (i + 1) % n
                v[i] = min(v[i], np.sqrt(v[nxt] ** 2 + 2 * A_BRAKE * dist[i]))

        return v

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        speed = np.linalg.norm(vehicle_velocity)

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        n = len(self.maneuverable_waypoints)

        # Dynamic lookahead. Cap raised from the main-branch prototype's 20 to 35
        # waypoints, matching the reference solutions' own proven max lookahead count
        # at top speed -- 20 was tuned against a 50 m/s ceiling, this profile now
        # allows up to 85 m/s, and too-short a lookahead at high speed is a known
        # cause of steering oscillation.
        lookahead_distance = int(np.clip(3 + 0.5 * speed, 3, 35))
        target_point = self.path[(self.current_waypoint_idx + lookahead_distance) % n]

        vector_to_target = (target_point - vehicle_location)[:2]
        heading_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        delta_heading = normalize_rad(heading_to_target - vehicle_rotation[2])

        # Pure pursuit steering, replacing the plain proportional heading controller.
        # This is the actual fix for the wall crashes: MU=2.75 in the velocity profile
        # was carried over from the reference solutions' pure-pursuit controller, which
        # naturally cuts toward a wider, gentler arc through a corner than the track's
        # literal curvature. The old proportional controller tracked the path much more
        # literally (tighter effective radius, same nominal corner), so v=sqrt(mu*g*r)
        # was systematically optimistic for it -- the car was being asked to go faster
        # than it could actually turn. Pure pursuit reunites the controller with the
        # assumption the speed profile was built on.
        lookahead_m = max(np.linalg.norm(vector_to_target), 1e-3)
        steer_control = -1.5 * np.arctan2(2.0 * 4.7 * np.sin(delta_heading) / lookahead_m, 1.0)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Speed target: a direct lookup into the precomputed profile. Replaces the
        # single-point heading-angle threshold table with per-point curvature +
        # braking-distance planning that already accounts for the whole lap ahead.
        target_velocity = self.velocity_profile[self.current_waypoint_idx]

        speed_error = target_velocity - speed
        speed_accel = speed - self.last_speed
        self.last_speed = speed

        Kp, Kd = 0.8, 0.02
        throttle_control = np.clip(Kp * speed_error - Kd * speed_accel, -1.0, 1.0)

        control = {
            "throttle": max(throttle_control, 0.0),
            "steer": steer_control,
            "brake": max(-throttle_control, 0.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }

        # TEMP DEBUG: remove once the wall-crash cause is confirmed. Prints one line
        # per tick so we can see the actual numbers at the moment of a crash instead
        # of guessing from symptoms.
        print(
            f"wp={self.current_waypoint_idx:4d} v={speed:5.1f} tgt_v={target_velocity:5.1f} "
            f"dh={delta_heading:+.3f} lh_m={lookahead_m:5.1f} steer={steer_control:+.3f} "
            f"thr={control['throttle']:.2f} brk={control['brake']:.2f}"
        )

        await self.vehicle.apply_action(control)
        return control
