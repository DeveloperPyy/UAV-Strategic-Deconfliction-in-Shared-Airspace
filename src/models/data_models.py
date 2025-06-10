# src/models/data_models.py
import math


class Waypoint:
    """
    Represents a single point in 3D space and time.
    """

    def __init__(self, x: float, y: float, z: float = 0.0, timestamp: float = None):
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = timestamp

    def to_tuple(self):
        """Returns the waypoint as a (x, y, z, timestamp) tuple."""
        return self.x, self.y, self.z, self.timestamp

    def distance_to(self, other_waypoint) -> float:
        """Calculates the Euclidean distance to another waypoint."""
        return math.sqrt(
            (self.x - other_waypoint.x) ** 2 +
            (self.y - other_waypoint.y) ** 2 +
            (self.z - other_waypoint.z) ** 2
        )

    def __repr__(self):
        return f"W(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f} @ t={self.timestamp:.2f})"


class DroneMission:
    """
    Represents a single drone's mission, defined by a sequence of waypoints
    and an overall time window.
    """

    def __init__(self, drone_id: str, waypoints: list[Waypoint],
                 mission_start_time: float = None, mission_end_time: float = None):
        self.drone_id = drone_id
        # Ensure waypoints are sorted by timestamp for correct interpolation
        self.waypoints = sorted(waypoints, key=lambda w: w.timestamp if w.timestamp is not None else float('inf'))

        # Overall mission time window as per requirement
        self.mission_start_time = mission_start_time
        self.mission_end_time = mission_end_time

        self.trajectory_points: list[Waypoint] = []  # Stores interpolated points (x,y,z,t)

    def generate_interpolated_trajectory(self, time_step: float = 1.0):
        """
        Generates a series of interpolated Waypoint objects representing the drone's trajectory
        at fixed time intervals.
        Each waypoint in the mission definition MUST have a timestamp.
        """
        if not self.waypoints:
            self.trajectory_points = []
            return

        for wp in self.waypoints:
            if wp.timestamp is None:
                raise ValueError(
                    f"Waypoint {wp} for drone {self.drone_id} does not have a timestamp. All waypoints must have "
                    f"timestamps for trajectory generation.")

        self.trajectory_points = []
        # Add the first waypoint
        self.trajectory_points.append(self.waypoints[0])

        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]

            segment_duration = wp2.timestamp - wp1.timestamp
            if segment_duration <= 0:
                # Handle cases where waypoints are at the same time or out of order
                # For now, just skip to avoid infinite loop / division by zero.
                # In a real system, this might be an error or indicate hovering.
                if segment_duration == 0:
                    # If waypoints are at same time, consider it a hover or direct jump
                    self.trajectory_points.append(wp2)
                continue

            num_steps = max(1, int(segment_duration / time_step))

            # Generate points for the segment
            for step in range(1, num_steps + 1):  # Start from 1 as 0 is wp1
                t_ratio = step / num_steps

                interp_x = wp1.x + (wp2.x - wp1.x) * t_ratio
                interp_y = wp1.y + (wp2.y - wp1.y) * t_ratio
                interp_z = wp1.z + (wp2.z - wp1.z) * t_ratio
                interp_t = wp1.timestamp + segment_duration * t_ratio

                # Add the interpolated point
                self.trajectory_points.append(Waypoint(interp_x, interp_y, interp_z, interp_t))

        # After generating all points, sort again to ensure perfect time order
        # (might not be strictly necessary if logic is perfect, but good for safety)
        self.trajectory_points = sorted(self.trajectory_points, key=lambda w: w.timestamp)

        # Filter points based on the overall mission time window if specified
        if self.mission_start_time is not None and self.mission_end_time is not None:
            self.trajectory_points = [
                wp for wp in self.trajectory_points
                if self.mission_start_time <= wp.timestamp <= self.mission_end_time
            ]

    def get_position_at_time(self, query_time: float) -> Waypoint | None:
        """
        Returns the interpolated Waypoint object at a specific query_time.
        Assumes trajectory_points has been generated and is sorted by time.
        If query_time is outside the mission's actual trajectory time bounds,
        it returns the closest known point (start or end) or None if no trajectory.
        """
        if not self.trajectory_points:
            return None

        # Check if query_time is before the first point
        if query_time < self.trajectory_points[0].timestamp:
            return self.trajectory_points[0]  # Drone hasn't started or is at start

        # Check if query_time is after the last point
        if query_time > self.trajectory_points[-1].timestamp:
            return self.trajectory_points[-1]  # Drone has finished or is at end

        # Find the two closest trajectory points that bracket the query_time
        # Linear scan for simplicity, binary search for performance with many points
        for i in range(len(self.trajectory_points) - 1):
            wp1 = self.trajectory_points[i]
            wp2 = self.trajectory_points[i + 1]

            if wp1.timestamp <= query_time <= wp2.timestamp:
                if wp1.timestamp == wp2.timestamp:
                    return wp1  # If points are at the same time, return the first one

                # Linear interpolation
                t_ratio = (query_time - wp1.timestamp) / (wp2.timestamp - wp1.timestamp)
                interp_x = wp1.x + (wp2.x - wp1.x) * t_ratio
                interp_y = wp1.y + (wp2.y - wp1.y) * t_ratio
                interp_z = wp1.z + (wp2.z - wp1.z) * t_ratio
                return Waypoint(interp_x, interp_y, interp_z, query_time)

        # Should not be reached if checks for before/after are correct and trajectory_points is sorted
        return None

    def get_actual_mission_time_range(self) -> tuple[float, float] | tuple[None, None]:
        """
        Returns the actual start and end timestamps covered by the generated trajectory points.
        """
        if not self.trajectory_points:
            return None, None
        return self.trajectory_points[0].timestamp, self.trajectory_points[-1].timestamp
