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
        self.print_counter = 0
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

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

        # Debug dashboard setup: print every N steps instead of every tick
        self.step_count = 0
        self.debug_print_interval = 20  # print roughly every 20 world steps


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
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
         # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        # waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 3) % len(self.maneuverable_waypoints)]
        # ----------------------------------------------------------
        # Dynamic Lookahead
        # ----------------------------------------------------------
        # Minimum number of waypoints to look ahead
        base_lookahead = 3
        # Controls how much the lookahead increases with speed
        k_v = 0.5
        # Calculate lookahead based on current speed
        lookahead_distance = int(base_lookahead + k_v * vehicle_velocity_norm)
        # Keep it between 3 and 20 waypoints
        lookahead_distance = np.clip(lookahead_distance, 3, 20)
        # Select the target waypoint
        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + lookahead_distance)
            % len(self.maneuverable_waypoints)
        ]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

# --- Curvature-aware target speed ---
        # Look further down the track than the steering target to estimate how
        # sharply the road curves ahead. Straight ahead -> high target speed.
        # Sharp turn ahead -> low target speed, so we brake before we reach it.
        lookahead_distance = 15  # waypoints further out than the steering target
        far_waypoint = self.maneuverable_waypoints[
            (self.current_waypoint_idx + 3 + lookahead_distance) % len(self.maneuverable_waypoints)
        ]
        vector_to_far_waypoint = (far_waypoint.location - waypoint_to_follow.location)[:2]
        heading_to_far_waypoint = np.arctan2(vector_to_far_waypoint[1], vector_to_far_waypoint[0])

        # How much the road direction changes between the near target and the far target.
        # ~0 rad = straight, larger = sharper corner ahead.
        curvature_angle = abs(normalize_rad(heading_to_far_waypoint - heading_to_waypoint))

        # Map curvature angle to a target speed: straight -> max_speed, sharp corner -> min_speed
        max_speed = 40.0   # m/s target on straights
        min_speed = 12.0   # m/s target for sharp corners
        curvature_angle_at_min_speed = 0.6  # radians (~34 degrees) considered "sharp"
        target_speed = np.interp(
            curvature_angle,
            [0.0, curvature_angle_at_min_speed],
            [max_speed, min_speed]
        )

        # Proportional controller to control the vehicle's speed towards the dynamic target
        throttle_control = 0.05 * (target_speed - vehicle_velocity_norm)




        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        # Debug dashboard: print every `debug_print_interval` steps to avoid flooding the console
        self.print_counter += 1

        if self.print_counter % 20 == 0:

            target_waypoint_idx = (
                self.current_waypoint_idx + lookahead_distance
            ) % len(self.maneuverable_waypoints)

            print("\n" + "=" * 70)
            print("           DYNAMIC LOOKAHEAD DEBUG")
            print("=" * 70)

            print(f"Current Speed        : {vehicle_velocity_norm:.2f} m/s")
            print(f"Current Speed        : {vehicle_velocity_norm*3.6:.2f} km/h")

            print()

            print(f"Current Waypoint     : {self.current_waypoint_idx}")
            print(f"Lookahead Distance   : {lookahead_distance}")
            print(f"Target Waypoint      : {target_waypoint_idx}")

            print()

            print(f"Current Steering     : {control['steer']:.3f}")
            print(f"Throttle             : {control['throttle']:.3f}")

            print("=" * 70)

        await self.vehicle.apply_action(control)
        return control
