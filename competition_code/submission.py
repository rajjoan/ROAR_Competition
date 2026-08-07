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
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.
        print("initialize() started", flush=True)

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # Smoothed path (triangle filter), used both as the steering target line and
        # as the curvature source for the velocity profile below.
        n = len(self.maneuverable_waypoints)
        self.path = [
            0.2 * self.maneuverable_waypoints[(i - 1) % n].location
            + 0.6 * self.maneuverable_waypoints[i].location
            + 0.2 * self.maneuverable_waypoints[(i + 1) % n].location
            for i in range(n)
        ]
        xy = [p[:2] for p in self.path]

        # Precompute a target speed (m/s) for every waypoint, once, offline:
        #   1. curvature-limited speed at each point (v = sqrt(mu*g*r)), via a
        #      3-point circle fit (Menger radius)
        #   2. a sliding-window MIN filter so one noisy curvature reading can't
        #      spike the target speed upward in the middle of a real corner
        #   3. a backward pass so braking for any corner starts as early as it
        #      actually needs to, not just within a fixed lookahead window
        # MU=2.75 is reused from the reference solutions' own proven "doesn't spin
        # out anywhere on this track" default. A_BRAKE=14.0 m/s^2 is back-derived
        # from the reference ThrottleController's braking formula (a=170..200 in
        # its km/h-based equation, /12.96 to convert to m/s^2). Both are still only
        # carried over from a different controller, not measured against this one.
        MU = 2.75
        G = 9.81
        V_MIN, V_MAX = 15.0, 85.0
        A_BRAKE = 14.0
        SMOOTH_WINDOW = 3

        def radius(p1, p2, p3, max_radius=10000.0):
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

        v_curv = np.array([
            np.clip(np.sqrt(MU * G * radius(xy[(i - 2) % n], xy[i], xy[(i + 2) % n])), V_MIN, V_MAX)
            for i in range(n)
        ])
        v_curv = np.array([
            min(v_curv[(i + off) % n] for off in range(-SMOOTH_WINDOW, SMOOTH_WINDOW + 1))
            for i in range(n)
        ])
        dist = np.array([np.linalg.norm(xy[(i + 1) % n] - xy[i]) for i in range(n)])

        v = v_curv.copy()
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                nxt = (i + 1) % n
                v[i] = min(v[i], np.sqrt(v[nxt] ** 2 + 2 * A_BRAKE * dist[i]))
        self.velocity_profile = v

        self.last_speed = 0.0
        self.stuck_ticks = 0
        self.stuck_override_ticks = 0
        print("initialize() finished", flush=True)

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        speed = np.linalg.norm(vehicle_velocity)

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        n = len(self.maneuverable_waypoints)

        # Dynamic lookahead (3 to 35 waypoints, scales with speed), then pure
        # pursuit steering toward that lookahead point on the smoothed path.
        lookahead_distance = int(np.clip(3 + 0.5 * speed, 3, 35))
        target_point = self.path[(self.current_waypoint_idx + lookahead_distance) % n]

        vector_to_target = (target_point - vehicle_location)[:2]
        heading_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        delta_heading = normalize_rad(heading_to_target - vehicle_rotation[2])

        lookahead_m = max(np.linalg.norm(vector_to_target), 1e-3)
        steer_control = -1.5 * np.arctan2(2.0 * 4.7 * np.sin(delta_heading) / lookahead_m, 1.0)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Speed target: a direct lookup into the precomputed profile.
        target_velocity = self.velocity_profile[self.current_waypoint_idx]

        speed_error = target_velocity - speed
        speed_accel = speed - self.last_speed
        self.last_speed = speed

        Kp, Kd = 0.8, 0.02
        throttle_control = np.clip(Kp * speed_error - Kd * speed_accel, -1.0, 1.0)

        # Stuck detection: if throttle is meaningfully open but speed isn't rising
        # for several ticks in a row WHILE ALREADY MOVING, override to full brake
        # instead of continuing to push into whatever it's wedged against.
        #
        # BUG FIXED HERE: the previous version didn't gate on speed > 0, so it also
        # triggered on any normal standing start / post-collision respawn (both have
        # speed~0 with high commanded throttle for the first few ticks, indistinguishable
        # from actually being stuck). Once triggered it forced throttle=-1.0 forever,
        # which guarantees speed stays 0 forever, which keeps the trigger condition
        # true forever -- a permanent deadlock. Confirmed via debug log: car frozen at
        # v=0.0, thr=0.00, brk=1.00, unchanged for 100+ ticks after a respawn.
        # Fixed with: (1) only counts as "stuck" above a real moving speed, ruling out
        # standing starts/respawns entirely; (2) the override now auto-releases after
        # a fixed number of ticks regardless, so even an unforeseen edge case can't
        # lock forever.
        STUCK_MIN_SPEED = 5.0
        STUCK_TRIGGER_TICKS = 5
        STUCK_OVERRIDE_TICKS = 15

        if throttle_control > 0.5 and speed_accel <= 0.05 and speed > STUCK_MIN_SPEED:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        if self.stuck_ticks > STUCK_TRIGGER_TICKS:
            self.stuck_override_ticks += 1
            if self.stuck_override_ticks <= STUCK_OVERRIDE_TICKS:
                throttle_control = -1.0
            else:
                self.stuck_ticks = 0
                self.stuck_override_ticks = 0
        else:
            self.stuck_override_ticks = 0

        control = {
            "throttle": max(throttle_control, 0.0),
            "steer": steer_control,
            "brake": max(-throttle_control, 0.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }

        # TEMP DEBUG: remove once the current issue is confirmed resolved.
        print(
            f"wp={self.current_waypoint_idx:4d} v={speed:5.1f} tgt_v={target_velocity:5.1f} "
            f"dh={delta_heading:+.3f} lh_m={lookahead_m:5.1f} steer={steer_control:+.3f} "
            f"thr={control['throttle']:.2f} brk={control['brake']:.2f}",
            flush=True
        )

        await self.vehicle.apply_action(control)
        return control
