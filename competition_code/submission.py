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
        # For example, you can compute the path to the first waypoint.        # Receive location, rotation and velocity data 
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
        # SPEED VS TURN_AMOUNT SLOPE FIX (v2): the single "if turn_amount > 0.35:
        # cap 42" patch from the previous commit still had a cliff -- both new crash
        # logs showed turn_amount at 0.349, just under the cutoff, releasing straight
        # to the uncapped formula (54.7) instead of easing down. A threshold just
        # moves the cliff, it doesn't remove it.
        #
        # Real fix: steepen the formula's own slope so it falls off continuously and
        # lands on the same already-proven safe values the prediction_turn ladder
        # below uses, with no separate cliff needed. Solved 62 - K*0.35 = 42 for K
        # (the exact turn_amount/target_velocity pair from the crash logs) -> K=57.
        # Checked against the other ladder anchors: turn=0.5 -> 33.5 (ladder: 33),
        # turn=0.7 -> 22.1 (ladder: 28, now more conservative, safe direction),
        # turn=0.9 -> clipped to the 15 floor (ladder: 24, also more conservative).
        # Slightly over-cautious at the sharpest end, which is the right side to
        # err on for now rather than risk a third crash at the same corner.
        target_velocity = 62 - (57 * turn_amount)

		# Predictive Braking
        if prediction_turn > 0.90: #0.75
            target_velocity = min(target_velocity, 24) #19
        elif prediction_turn > 0.70:
            target_velocity = min(target_velocity, 28)#23
        elif prediction_turn > 0.50:
            target_velocity = min(target_velocity, 33)#23
        elif prediction_turn > 0.35: #0.25
            target_velocity = min(target_velocity, 42)#30

        # CORNER-SPECIFIC OVERRIDE: waypoint ~500 crashed 3 times in a row despite
        # 3 separate, confirmed-correct fixes to the general formula (the cliff, the
        # slope, and the 45% braking-authority cap) -- each fix demonstrably worked
        # (verified via debug log each time) and it still crashed here, meaning the
        # ceiling isn't in those constants anymore, it's specific to this corner.
        # Rather than keep tuning the global formula against this one spot (which
        # risks either another crash here or over-conservatism everywhere else),
        # give it its own separate, explicit cap. Range covers where turn_amount was
        # observed building (waypoint ~491) through where the crashes happened
        # (500-502), with margin on both sides for reaction time.
        # 38 m/s is deliberately conservative: below the already-proven-safe 42
        # ladder value used elsewhere for similar turn_amount, and below the
        # original 50 m/s ceiling this corner was never tested past. After 3 failed
        # attempts here, prioritizing certainty over squeezing out a few more m/s.
        if 480 <= self.current_waypoint_idx <= 510:
            target_velocity = min(target_velocity, 38)

        target_velocity = np.clip(target_velocity,15,70)

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

        target_velocity = np.clip(target_velocity, 15, 70)

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
        # Braking authority was hard-capped at 45% (-0.45), which never mattered at
        # main's original 50 m/s ceiling (speed errors were always small enough for
        # 45% brake to be sufficient). Confirmed via debug log: brk pinned at exactly
        # 0.45 for 10+ consecutive ticks while v stayed 15-24 m/s above tgt_v the
        # whole way through the corner and never caught up -- not enough braking
        # force to shed the larger speed deficits the raised cap now creates. Full
        # range matches what every other controller in this repo already uses.
        throttle_control = np.clip(throttle_control, -1.0, 1.0)
        control = {
                "throttle": max(throttle_control, 0.0),
                "steer": steer_control,
                "brake": max(-throttle_control, 0.0),
                "hand_brake": 0.0,
                "reverse": 0,
                "target_gear": 0,
            }

        # TEMP DEBUG: remove once the waypoint-500 corner is diagnosed. main has no
        # existing telemetry, and the last crash log there (position only) couldn't
        # tell us whether this is overspeed-into-the-corner or too-late-braking.
        print(
            f"wp={self.current_waypoint_idx:4d} v={vehicle_velocity_norm:5.1f} "
            f"tgt_v={target_velocity:5.1f} turn={turn_amount:.3f} pred_turn={prediction_turn:.3f} "
            f"thr={control['throttle']:.2f} brk={control['brake']:.2f}",
            flush=True
        )

        await self.vehicle.apply_action(control)
        return control
