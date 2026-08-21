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
        self.frame = 0
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # Build a smoothed racing line
        self.optimized_waypoints=[]
        n=len(self.maneuverable_waypoints)
        for i in range(n):
            p=self.maneuverable_waypoints[(i-1)%n]
            c=self.maneuverable_waypoints[i]
            nx=self.maneuverable_waypoints[(i+1)%n]
            wp=roar_py_interface.RoarPyWaypoint(
                location=0.2*p.location+0.6*c.location+0.2*nx.location,
                roll_pitch_yaw=c.roll_pitch_yaw,
                lane_width=c.lane_width
            )
            self.optimized_waypoints.append(wp)



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
        ## ----------------------------------------------------------
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
        waypoint_to_follow = self.optimized_waypoints[
            (self.current_waypoint_idx + lookahead_distance)
            % len(self.maneuverable_waypoints)
        ]
        

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
		
# ----------------------------------------------------------
# Predictive Intelligent Braking
# ----------------------------------------------------------
        prediction_offset = 17
        prediction_distance = lookahead_distance + prediction_offset

        prediction_waypoint = self.maneuverable_waypoints[
			(self.current_waypoint_idx + prediction_distance)
			% len(self.maneuverable_waypoints)]
			
			
        prediction_vector = (
			prediction_waypoint.location - vehicle_location
			)[:2]

        prediction_heading = np.arctan2(
		prediction_vector[1],
		prediction_vector[0]
		)

        prediction_delta_heading = normalize_rad(
		prediction_heading - vehicle_rotation[2]
		)

        prediction_turn = abs(prediction_delta_heading)


# ----------------------------------------------------------
# Adaptive Target Speed
# ----------------------------------------------------------

        turn_amount = abs(delta_heading)
        target_velocity = 48 - (21 * turn_amount)   

		# Predictive Braking 
        if prediction_turn > 0.90: #0.75
            target_velocity = min(target_velocity, 24) #19
        elif prediction_turn > 0.70:
            target_velocity = min(target_velocity, 28)#23
        elif prediction_turn > 0.50:
            target_velocity = min(target_velocity, 33)#23
        elif prediction_turn > 0.35: #0.25
            target_velocity = min(target_velocity, 42)#30
 
        target_velocity = np.clip(target_velocity,15,50)

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Proportional controller to control the vehicle's speed towards 40 m/s
        #throttle_control = 0.05 * (20 - vehicle_velocity_norm)  

# ----------------------------------------------------------
# Aggressive Throttle Recovery
# ----------------------------------------------------------

        recovery_speed = 8

        if turn_amount < 0.15 and prediction_turn < 0.1:
            target_velocity += recovery_speed
 
        target_velocity = np.clip(target_velocity, 15, 50)

# ----------------------------------------------------------
# PD Speed Controller
# ----------------------------------------------------------

# Desired cruising speed
#        target_velocity = 30.0   # 30 m/s ≈ 108 km/h

# P Term: How far are we from the target speed?
        speed_error = target_velocity - vehicle_velocity_norm

# D Term: How much has the speed changed since the last frame?
        last_speed = getattr(self, "last_speed", vehicle_velocity_norm)
        speed_acceleration = vehicle_velocity_norm - last_speed

# Save current speed for the next frame
        self.last_speed = vehicle_velocity_norm 

# Controller gains
        Kp = 0.8
        Kd = 0.02

# PD Controller
        #throttle_control = (Kp * speed_error) - (Kd * speed_acceleration)
        throttle_control = (Kp * speed_error) - (Kd * speed_acceleration)
        throttle_control = np.clip(throttle_control, -0.45, 1.0)
        control = {
                "throttle": max(throttle_control, 0.0),
                "steer": steer_control,
                "brake": max(-throttle_control, 0.0),
                "hand_brake": 0.0,
                "reverse": 0,
                "target_gear": 0,
            }

        await self.vehicle.apply_action(control)
        return control