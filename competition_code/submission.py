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
        #waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 3) % len(self.maneuverable_waypoints)]

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


        turn_amount = abs(delta_heading)
        target_velocity = 32 - (25 * turn_amount)
        target_velocity = np.clip(target_velocity, 20, 32)

        # if turn_amount < 0.15:
        #     target_velocity = 35.0
        # elif turn_amount < 0.3:
        #     target_velocity = 25.0
        # elif turn_amount < 0.6:
        #     target_velocity = 20.0
        # else:
        #     target_velocity = 15.0

        # Proportional controllerto steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

# ----------------------------------------------------------
# PD Speed Controller
# ----------------------------------------------------------

# Desired cruising speed
  #      target_velocity = 27.0   # 40 m/s ≈ 144 km/h 

# P Term: How far are we from the target speed? 
        speed_error = target_velocity - vehicle_velocity_norm

# D Term: How much has the speed changed since the last frame?
        last_speed = getattr(self, "last_speed", vehicle_velocity_norm)
        speed_acceleration = vehicle_velocity_norm - last_speed

# Save current speed for the next frame
        self.last_speed = vehicle_velocity_norm

# Controller gains
        Kp = 0.6
        Kd = 0.1 

# PD Controller
        throttle_control = (Kp * speed_error) - (Kd * speed_acceleration)  
 
        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }

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
