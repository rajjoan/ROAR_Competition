import numpy as np
import math


def normalize_to_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


class LatController:
    WHEELBASE = 4.7
    K = 2.0
    SOFT = 10.0

    def run(self, vehicle_location, vehicle_rotation,
            nearest_waypoint_location, waypoint_to_follow_location,
            current_waypoint_idx, current_speed_ms):
        vehicle_yaw = float(vehicle_rotation[2])
        path_vec = np.array(waypoint_to_follow_location[:2]) - np.array(nearest_waypoint_location[:2])
        pv_norm = float(np.linalg.norm(path_vec))
        if pv_norm < 1e-3:
            return 0.0, ""
        path_heading = math.atan2(path_vec[1], path_vec[0])
        heading_error = float(np.clip(
            normalize_to_pi(vehicle_yaw - path_heading),
            -math.pi / 2, math.pi / 2
        ))
        front_axle = np.array([
            vehicle_location[0] + (self.WHEELBASE / 2.0) * math.cos(vehicle_yaw),
            vehicle_location[1] + (self.WHEELBASE / 2.0) * math.sin(vehicle_yaw),
        ])
        path_normal = np.array([-math.sin(path_heading), math.cos(path_heading)])
        delta = front_axle - np.array(nearest_waypoint_location[:2])
        cross_track_error = float(np.dot(delta, path_normal))
        speed = max(float(current_speed_ms), 0.1)
        cte_term = math.atan2(self.K * cross_track_error, speed + self.SOFT)
        k_heading = max(0.5, self.SOFT / (speed + self.SOFT))
        steering_command = k_heading * heading_error + cte_term
        if speed < 5.0:
            steering_command = float(np.clip(steering_command, -0.3, 0.3))
        elif speed < 15.0:
            steering_command = float(np.clip(steering_command, -0.8, 0.8))
        debug_str = (f"stanley[{current_waypoint_idx}]: hdg={heading_error:.3f} "
                     f"cte={cross_track_error:.3f} cmd={steering_command:.3f}")
        return float(steering_command), debug_str
