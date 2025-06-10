# src/deconfliction/conflict_detector.py

from typing import List, Tuple, Optional
from src.models.data_models import Waypoint, DroneMission
import math


class Conflict:
    """
    Represents a detected conflict between two drones.
    Encapsulates all necessary information about the conflict for reporting.
    """

    def __init__(self, time_of_conflict: float,
                 primary_drone_pos: Waypoint,
                 conflicting_drone_id: str,
                 conflicting_drone_pos: Waypoint,
                 safety_buffer: float):
        self.time_of_conflict = time_of_conflict
        self.primary_drone_pos = primary_drone_pos
        self.conflicting_drone_id = conflicting_drone_id
        self.conflicting_drone_pos = conflicting_drone_pos
        self.safety_buffer = safety_buffer
        # Calculate the actual distance at the moment of conflict
        self.distance_at_conflict = primary_drone_pos.distance_to(conflicting_drone_pos)

    def __repr__(self):
        """Provides a user-friendly string representation of the conflict."""
        return (f"Conflict at t={self.time_of_conflict:.2f}:\n"
                f"  Primary Drone (ID: Primary) at ({self.primary_drone_pos.x:.2f},{self.primary_drone_pos.y:.2f},{self.primary_drone_pos.z:.2f})\n"
                f"  Collided with Drone '{self.conflicting_drone_id}' at ({self.conflicting_drone_pos.x:.2f},{self.conflicting_drone_pos.y:.2f},{self.conflicting_drone_pos.z:.2f})\n"
                f"  Distance: {self.distance_at_conflict:.2f} (Required Safety: {self.safety_buffer:.2f})")

    def get_conflict_details(self) -> dict:
        """Returns conflict details as a dictionary, useful for structured output or logging."""
        return {
            "time": self.time_of_conflict,
            "primary_drone_position": self.primary_drone_pos.to_tuple(),
            "conflicting_drone_id": self.conflicting_drone_id,
            "conflicting_drone_position": self.conflicting_drone_pos.to_tuple(),
            "distance_at_conflict": self.distance_at_conflict,
            "safety_buffer_applied": self.safety_buffer
        }


def check_for_conflicts(
        primary_mission: DroneMission,
        simulated_schedules: List[DroneMission],
        safety_buffer: float,
        time_step: float = 1.0
) -> Tuple[str, Optional[List[Conflict]]]:
    detected_conflicts: List[Conflict] = []

    # Ensure trajectories are generated for all drones.
    if not primary_mission.trajectory_points:
        primary_mission.generate_interpolated_trajectory(time_step)

    for sim_mission in simulated_schedules:
        if not sim_mission.trajectory_points:
            sim_mission.generate_interpolated_trajectory(time_step)

    # Determine the effective time window for conflict checking for the PRIMARY drone.
    primary_actual_start_t, primary_actual_end_t = primary_mission.get_actual_mission_time_range()

    if primary_actual_start_t is None or primary_actual_end_t is None:
        return "clear", None

    # Use the primary mission's defined time window if provided, otherwise default to actual flight time.
    # This defines the "query window" for the primary drone.
    query_start_time = primary_mission.mission_start_time if primary_mission.mission_start_time is not None else primary_actual_start_t
    query_end_time = primary_mission.mission_end_time if primary_mission.mission_end_time is not None else primary_actual_end_t

    # Adjust query window to be within the actual generated trajectory bounds
    query_start_time = max(query_start_time, primary_actual_start_t)
    query_end_time = min(query_end_time, primary_actual_end_t)

    # Iterate through time steps within the primary drone's effective checking window
    current_time = query_start_time
    while current_time <= query_end_time:
        # Get primary drone's position at current_time
        primary_pos = primary_mission.get_position_at_time(current_time)

        if primary_pos is None:  # This should generally not happen if `current_time` is within bounds
            current_time += time_step
            continue

        for sim_mission in simulated_schedules:
            sim_actual_start_t, sim_actual_end_t = sim_mission.get_actual_mission_time_range()

            # Check if the current_time falls within the simulated drone's *actual* flight time range.
            # If the simulated drone is not actively flying at this moment, it cannot cause a conflict.
            if sim_actual_start_t is None or sim_actual_end_t is None or \
                    not (sim_actual_start_t <= current_time <= sim_actual_end_t):
                continue  # Simulated drone is not active at this time, skip conflict check for it.

            sim_pos = sim_mission.get_position_at_time(current_time)

            # sim_pos should not be None here due to the above check, but keep for robustness
            if sim_pos is not None:
                distance = primary_pos.distance_to(sim_pos)

                if distance < safety_buffer:
                    conflict = Conflict(
                        time_of_conflict=current_time,
                        primary_drone_pos=primary_pos,
                        conflicting_drone_id=sim_mission.drone_id,
                        conflicting_drone_pos=sim_pos,
                        safety_buffer=safety_buffer
                    )
                    detected_conflicts.append(conflict)

        current_time += time_step

    if detected_conflicts:
        return "conflict detected", detected_conflicts
    else:
        return "clear", None
