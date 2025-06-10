# src/main.py

from src.simulation import ScenarioGenerator
from src.deconfliction import check_for_conflicts, Conflict
from src.visualization import Plotter
from src.models.data_models import Waypoint, DroneMission

import os
import sys
import json
from datetime import datetime
import numpy as np  # Import numpy for arange

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_deconfliction_simulation(scenario_name: str,
                                 data_file: str = '../data/simulated_flights.json',
                                 output_media_dir: str = 'media/animations',
                                 output_report_dir: str = 'media/reports',
                                 output_plots_dir: str = 'media/plots'):
    """
    Runs a deconfliction simulation for a specified scenario, checks for conflicts,
    and generates visualizations and a conflict report.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Ensure output directories exist
    os.makedirs(output_media_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # 1. Load Scenario Data
    scenario_gen = ScenarioGenerator(data_file)
    primary_mission, simulated_missions = scenario_gen.get_scenario(scenario_name)
    safety_buffer = scenario_gen.get_global_safety_buffer()
    time_step = scenario_gen.get_global_time_step()

    print(f"Loaded scenario: '{scenario_name}'")
    print(f"Primary Drone Waypoints: {len(primary_mission.waypoints)}")
    print(f"Simulated Drones: {len(simulated_missions)}")
    print(f"Safety Buffer: {safety_buffer:.2f}, Time Step: {time_step:.2f}")

    # Prepare for report file output - unique filename with timestamp
    report_filename = os.path.join(output_report_dir,
                                   f"{scenario_name}_deconfliction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Collect content for the report file and terminal output
    report_lines = []

    report_lines.append(f"Deconfliction Report for Scenario: '{scenario_name}'")
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("-" * 60)
    report_lines.append(f"Primary Drone ID: {primary_mission.drone_id}")
    report_lines.append(f"Simulated Drones ({len(simulated_missions)}): {[d.drone_id for d in simulated_missions]}")
    report_lines.append(f"Safety Buffer: {safety_buffer:.2f} meters")
    report_lines.append(f"Time Step for Simulation: {time_step:.2f} seconds")
    report_lines.append("-" * 60)

    # --- NEW/MODIFIED: Calculate global time points and distances once ---
    all_traj_points_combined = list(primary_mission.trajectory_points)
    for sim_mission in simulated_missions:
        all_traj_points_combined.extend(sim_mission.trajectory_points)

    if not all_traj_points_combined:
        print("No trajectory points found for any drone. Skipping simulation and plotting.")
        report_lines.append("No trajectory points found for any drone. Skipping simulation and plotting.")
        with open(report_filename, 'w') as f:
            f.write("\n".join(report_lines))
        return

    min_time = min(wp.timestamp for wp in all_traj_points_combined if wp.timestamp is not None)
    max_time = max(wp.timestamp for wp in all_traj_points_combined if wp.timestamp is not None)

    # Ensure max_time covers the full mission range if drones have different end times
    effective_max_time_primary = primary_mission.get_actual_mission_time_range()[1]
    effective_max_time_sim = max((sm.get_actual_mission_time_range()[1] for sm in simulated_missions), default=min_time)
    effective_overall_max_time = max(effective_max_time_primary, effective_max_time_sim)

    plot_times = np.arange(min_time, effective_overall_max_time + time_step, time_step)

    distances_over_time = {sim_mission.drone_id: [] for sim_mission in simulated_missions}

    for current_time in plot_times:
        primary_pos = primary_mission.get_position_at_time(current_time)

        for sim_mission in simulated_missions:
            sim_pos = sim_mission.get_position_at_time(current_time)

            if primary_pos and sim_pos:
                distance = primary_pos.distance_to(sim_pos)
                distances_over_time[sim_mission.drone_id].append(distance)
            else:
                distances_over_time[sim_mission.drone_id].append(np.nan)  # Mark as NaN if drone not active

    # 2. Perform Deconfliction Check (this still returns discrete conflict points)
    status, conflicts = check_for_conflicts(
        primary_mission, simulated_missions, safety_buffer, time_step
    )

    # 3. Report Results to Terminal and File
    if status == "clear":
        terminal_message = f"DECONFLICTION STATUS: {status.upper()} - No conflicts detected."
        report_lines.append(terminal_message)
        print(terminal_message)
    else:
        terminal_message = f"DECONFLICTION STATUS: {status.upper()} - {len(conflicts)} conflict(s) detected!"
        report_lines.append(terminal_message)
        print(terminal_message)

        report_lines.append("\n--- Detected Conflicts ---")
        print("\n--- Detected Conflicts ---")
        for i, conflict in enumerate(conflicts):
            conflict_detail_str = (
                f"Conflict {i + 1}:\n"
                f"  Time of Conflict: {conflict.time_of_conflict:.2f} seconds\n"
                f"  Distance at Conflict: {conflict.distance_at_conflict:.2f} meters (Safety Buffer: {safety_buffer:.2f}m)\n"
                f"  Primary Drone ({conflict.primary_drone_pos.drone_id if hasattr(conflict.primary_drone_pos, 'drone_id') else 'N/A'})\n"
                f"    Position: (X={conflict.primary_drone_pos.x:.2f}, Y={conflict.primary_drone_pos.y:.2f}, Z={conflict.primary_drone_pos.z:.2f})\n"
                f"    Timestamp: {conflict.primary_drone_pos.timestamp:.2f}s\n"
                f"  Conflicting Drone ID: {conflict.conflicting_drone_id}\n"
                f"    Position: (X={conflict.conflicting_drone_pos.x:.2f}, Y={conflict.conflicting_drone_pos.y:.2f}, Z={conflict.conflicting_drone_pos.z:.2f})\n"
                f"    Timestamp: {conflict.conflicting_drone_pos.timestamp:.2f}s"
            )
            report_lines.append(conflict_detail_str)
            print(conflict_detail_str)
            if i < len(conflicts) - 1:
                print("-" * 30)
                report_lines.append("-" * 30)
        report_lines.append("-" * 60)

        # Save report to file
    try:
        with open(report_filename, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"Deconfliction report saved to: {report_filename}")
    except IOError as e:
        print(f"ERROR: Could not save report to {report_filename}. Reason: {e}")

    # 4. Visualize Results (Matplotlib GIF)
    print("Generating Matplotlib visualization (GIF)...")
    plotter = Plotter(output_media_dir)
    plotter.plot_scenario_animation(
        scenario_name,
        primary_mission,
        simulated_missions,
        conflicts,
        safety_buffer,
        time_step
    )
    print(f"Matplotlib visualization saved for scenario '{scenario_name}' in {output_media_dir}")

    # 5. Visualize Results (Plotly HTML)
    print("Generating Plotly visualization (HTML)...")
    plotter.plot_scenario_plotly_animation(
        scenario_name,
        primary_mission,
        simulated_missions,
        conflicts,
        safety_buffer,
        time_step
    )
    print(f"Plotly visualization saved for scenario '{scenario_name}' in {plotter.plotly_output_dir}")

    # 6. Visualize Results (Distance vs. Time Plot)
    print("Generating Distance vs. Time plot (PNG)...")
    plotter.plot_distance_vs_time(
        scenario_name,
        primary_mission,
        simulated_missions,
        conflicts,
        safety_buffer,
        plot_times,  # Pass pre-calculated
        distances_over_time  # Pass pre-calculated
    )
    print(f"Distance vs. Time plot saved for scenario '{scenario_name}' in {plotter.plots_output_dir}")

    # 7. Visualize Results (Temporal Conflict Timeline/Gantt Chart - NEW)
    print("Generating Temporal Conflict Timeline plot (PNG)...")
    plotter.plot_temporal_conflict_timeline(
        scenario_name,
        primary_mission.drone_id,  # Pass primary drone ID for label
        simulated_missions,
        safety_buffer,
        plot_times,
        distances_over_time
    )
    print(f"Temporal Conflict Timeline plot saved for scenario '{scenario_name}' in {plotter.plots_output_dir}")


if __name__ == "__main__":
    # Ensure top-level output directories exist
    os.makedirs('media', exist_ok=True)
    os.makedirs('media/animations', exist_ok=True)
    os.makedirs('media/reports', exist_ok=True)
    os.makedirs('media/plots', exist_ok=True)

    # Example: Run all scenarios defined in your JSON
    scenario_generator = ScenarioGenerator('../data/simulated_flights.json')
    all_scenario_names = scenario_generator.get_all_scenario_names()

    if not all_scenario_names:
        print("No scenarios found in data/simulated_flights.json. Please define some.")
    else:
        for name in all_scenario_names:
            run_deconfliction_simulation(name)

    print("\n--- All simulations complete ---")
