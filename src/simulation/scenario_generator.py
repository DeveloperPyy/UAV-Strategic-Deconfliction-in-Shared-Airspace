# src/simulation/scenario_generator.py

import json
from typing import List, Dict, Union, Tuple

from src.models.data_models import Waypoint, DroneMission


class ScenarioGenerator:
    """
    Handles loading drone mission data from a JSON file and generating
    DroneMission objects for different scenarios.
    """

    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.data: Dict = self._load_data()
        self.safety_buffer: float = self.data.get("safety_buffer", 5.0)  # Default if not in JSON
        self.time_step: float = self.data.get("time_step", 1.0)  # Default if not in JSON

    def _load_data(self) -> Dict:
        """Loads the JSON data from the specified file path."""
        try:
            with open(self.data_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {self.data_file_path}")

    def _parse_waypoints(self, raw_waypoints: List[Dict]) -> List[Waypoint]:
        """Parses a list of raw waypoint dictionaries into Waypoint objects."""
        waypoints = []
        for wp_data in raw_waypoints:
            # Ensure z is present for 4D extra credit, default to 0.0 if not
            waypoints.append(Waypoint(
                x=wp_data['x'],
                y=wp_data['y'],
                z=wp_data.get('z', 0.0),  # Get 'z' with default 0.0 if not present
                timestamp=wp_data['timestamp']
            ))
        return waypoints

    def get_scenario(self, scenario_name: str) -> Tuple[DroneMission, List[DroneMission]]:
        """
        Retrieves a specific scenario by name and parses it into DroneMission objects.

        Args:
            scenario_name: The name of the scenario to retrieve.

        Returns:
            A tuple containing (primary_drone_mission, list_of_simulated_drone_missions).
        """
        for scenario_data in self.data.get("scenarios", []):
            if scenario_data["scenario_name"] == scenario_name:
                # Parse primary drone mission
                primary_drone_data = scenario_data["primary_drone"]
                primary_waypoints = self._parse_waypoints(primary_drone_data["waypoints"])
                primary_mission = DroneMission(
                    drone_id=primary_drone_data["drone_id"],
                    waypoints=primary_waypoints,
                    mission_start_time=primary_drone_data.get("mission_start_time"),
                    mission_end_time=primary_drone_data.get("mission_end_time")
                )
                # Generate trajectory immediately upon loading
                primary_mission.generate_interpolated_trajectory(self.time_step)

                # Parse simulated drone missions
                simulated_missions = []
                for sim_drone_data in scenario_data["simulated_drones"]:
                    sim_waypoints = self._parse_waypoints(sim_drone_data["waypoints"])
                    sim_mission = DroneMission(
                        drone_id=sim_drone_data["drone_id"],
                        waypoints=sim_waypoints
                    )
                    # Generate trajectory immediately upon loading
                    sim_mission.generate_interpolated_trajectory(self.time_step)
                    simulated_missions.append(sim_mission)

                return primary_mission, simulated_missions

        raise ValueError(f"Scenario '{scenario_name}' not found in data file.")

    def get_all_scenario_names(self) -> List[str]:
        """Returns a list of all available scenario names."""
        return [s["scenario_name"] for s in self.data.get("scenarios", [])]

    def get_global_safety_buffer(self) -> float:
        """Returns the global safety buffer defined in the data file."""
        return self.safety_buffer

    def get_global_time_step(self) -> float:
        """Returns the global time step for trajectory generation and conflict checks."""
        return self.time_step
