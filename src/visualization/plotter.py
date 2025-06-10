# src/visualization/plotter.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from typing import List, Tuple, Optional, Dict
import numpy as np

import plotly.graph_objects as go
from plotly.offline import plot as py_plot

from src.models.data_models import Waypoint, DroneMission
from src.deconfliction import Conflict


class Plotter:
    """
    Handles visualization of drone trajectories and conflicts using Matplotlib and Plotly.
    Generates static plots, Matplotlib animations, and interactive Plotly animations.
    """

    def __init__(self, output_dir: str = 'media/animations'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.plotly_output_dir = os.path.join(output_dir, 'plotly_animations')
        os.makedirs(self.plotly_output_dir, exist_ok=True)
        self.plots_output_dir = os.path.join('media', 'plots')
        os.makedirs(self.plots_output_dir, exist_ok=True)

    def _get_all_waypoints(self, primary_mission: DroneMission,
                           simulated_missions: List[DroneMission]) -> List[Waypoint]:
        """Helper to collect all waypoints for plot limits."""
        all_waypoints = list(primary_mission.waypoints)
        for sim_mission in simulated_missions:
            all_waypoints.extend(sim_mission.waypoints)
        return all_waypoints

    def _get_all_trajectory_points(self, primary_mission: DroneMission,
                                   simulated_missions: List[DroneMission]) -> List[Waypoint]:
        """Helper to collect all interpolated trajectory points for determining time range."""
        all_traj_points = list(primary_mission.trajectory_points)
        for sim_mission in simulated_missions:
            all_traj_points.extend(sim_mission.trajectory_points)
        return all_traj_points

    def _generate_sphere_points(self, center_x: float, center_y: float, center_z: float, radius: float,
                                num_lines: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generates points for multiple wireframe circles to form a sphere.
        Returns a list of (xs, ys, zs) tuples, each representing a circle (meridian or parallel).
        These 1D arrays are suitable for go.Scatter3d(mode='lines').
        """
        sphere_lines = []

        # Parallels (horizontal circles)
        phi = np.linspace(0, np.pi, num_lines)
        theta = np.linspace(0, 2 * np.pi, 50)
        for p in phi:
            xs = center_x + radius * np.sin(p) * np.cos(theta)
            ys = center_y + radius * np.sin(p) * np.sin(theta)
            zs = center_z + radius * np.cos(p) * np.ones_like(theta)
            sphere_lines.append((xs, ys, zs))

        # Meridians (vertical circles)
        for t in np.linspace(0, 2 * np.pi, num_lines, endpoint=False):
            xs = center_x + radius * np.sin(phi) * np.cos(t)
            ys = center_y + radius * np.sin(phi) * np.sin(t)
            zs = center_z + radius * np.cos(phi)
            sphere_lines.append((xs, ys, zs))

        return sphere_lines

    # --- Existing Matplotlib animation function (no changes) ---
    def plot_scenario_animation(self,
                                scenario_name: str,
                                primary_mission: DroneMission,
                                simulated_missions: List[DroneMission],
                                conflicts: Optional[List[Conflict]],
                                safety_buffer: float,
                                time_step: float):
        """
        Generates an animated plot of the drone trajectories, safety buffers, and highlights conflicts
        using Matplotlib.
        """
        all_waypoints = self._get_all_waypoints(primary_mission, simulated_missions)

        # Determine plot limits based on all waypoints (x, y, z)
        min_x = min(wp.x for wp in all_waypoints) - (safety_buffer + 10)
        max_x = max(wp.x for wp in all_waypoints) + (safety_buffer + 10)
        min_y = min(wp.y for wp in all_waypoints) - (safety_buffer + 10)
        max_y = max(wp.y for wp in all_waypoints) + (safety_buffer + 10)
        min_z = min(wp.z for wp in all_waypoints) - (safety_buffer + 5)
        max_z = max(wp.z for wp in all_waypoints) + (safety_buffer + 5)

        # Determine the total time range for animation
        all_traj_points = self._get_all_trajectory_points(primary_mission, simulated_missions)
        if not all_traj_points:
            print("No trajectory points to animate.")
            return

        min_time = min(wp.timestamp for wp in all_traj_points if wp.timestamp is not None)
        max_time = max(wp.timestamp for wp in all_traj_points if wp.timestamp is not None)

        effective_max_time = max(primary_mission.get_actual_mission_time_range()[1],
                                 max(sm.get_actual_mission_time_range()[1] for sm in simulated_missions if
                                     sm.get_actual_mission_time_range()[1] is not None))

        current_frame_time = min_time
        frames = []
        while current_frame_time <= effective_max_time + time_step:
            frames.append(current_frame_time)
            current_frame_time += time_step

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot static elements: full trajectories
        ax.plot([wp.x for wp in primary_mission.trajectory_points],
                [wp.y for wp in primary_mission.trajectory_points],
                [wp.z for wp in primary_mission.trajectory_points],
                'b--', alpha=0.3, label='Primary Trajectory (Full)')

        for sim_mission in simulated_missions:
            ax.plot([wp.x for wp in sim_mission.trajectory_points],
                    [wp.y for wp in sim_mission.trajectory_points],
                    [wp.z for wp in sim_mission.trajectory_points],
                    'k:', alpha=0.2,
                    label='Simulated Trajectory (Full)' if sim_mission == simulated_missions[0] else "")

        # Initialize drone markers
        primary_marker, = ax.plot([], [], [], 'ro', markersize=8, label='Primary Drone')

        primary_sphere_wireframes = [ax.plot([], [], [], 'r--', alpha=0.2, linewidth=0.8)[0]
                                     for _ in range(
                self._generate_sphere_points(0, 0, 0, safety_buffer, num_lines=10).__len__())]

        sim_markers = []
        sim_sphere_wireframes_list = []
        sim_colors = plt.cm.viridis(np.linspace(0, 1, len(simulated_missions)))
        for i, sim_mission in enumerate(simulated_missions):
            marker, = ax.plot([], [], [], 'o', color=sim_colors[i], markersize=6,
                              label=f'Sim Drone {sim_mission.drone_id}')
            sim_markers.append(marker)
            sim_drone_sphere_wfs = [ax.plot([], [], [], '--', color=sim_colors[i], alpha=0.15, linewidth=0.6)[0]
                                    for _ in
                                    range(self._generate_sphere_points(0, 0, 0, safety_buffer, num_lines=10).__len__())]
            sim_sphere_wireframes_list.append(sim_drone_sphere_wfs)

        # Initialize conflict markers
        conflict_points = ax.plot([], [], [], 'rx', markersize=10, mew=2, label='Conflict Point')[0]

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Scenario: {scenario_name} - Time: {min_time:.2f}s")
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=20., azim=-45)

        # Cache conflicts for quick lookup
        time_indexed_conflicts = {}
        if conflicts:
            for conflict in conflicts:
                conflict_time_rounded = round(conflict.time_of_conflict / time_step) * time_step
                if conflict_time_rounded not in time_indexed_conflicts:
                    time_indexed_conflicts[conflict_time_rounded] = []
                time_indexed_conflicts[conflict_time_rounded].append(conflict)

        def update(frame_time):
            """Update function for the animation."""
            ax.set_title(f"Scenario: {scenario_name} - Time: {frame_time:.2f}s")
            artists = []

            # Update primary drone position and buffer
            primary_pos = primary_mission.get_position_at_time(frame_time)
            if primary_pos:
                primary_marker.set_data_3d([primary_pos.x], [primary_pos.y], [primary_pos.z])
                sphere_lines_data = self._generate_sphere_points(primary_pos.x, primary_pos.y, primary_pos.z,
                                                                 safety_buffer)
                for i, (sx, sy, sz) in enumerate(sphere_lines_data):
                    primary_sphere_wireframes[i].set_data_3d(sx, sy, sz)
            else:
                primary_marker.set_data_3d([], [], [])
                for wf in primary_sphere_wireframes:
                    wf.set_data_3d([], [], [])
            artists.append(primary_marker)
            artists.extend(primary_sphere_wireframes)

            # Update simulated drone positions and buffers
            for i, sim_mission in enumerate(simulated_missions):
                sim_pos = sim_mission.get_position_at_time(frame_time)
                if sim_pos:
                    sim_markers[i].set_data_3d([sim_pos.x], [sim_pos.y], [sim_pos.z])
                    sphere_lines_data = self._generate_sphere_points(sim_pos.x, sim_pos.y, sim_pos.z, safety_buffer)
                    for j, (sx, sy, sz) in enumerate(sphere_lines_data):
                        sim_sphere_wireframes_list[i][j].set_data_3d(sx, sy, sz)
                else:
                    sim_markers[i].set_data_3d([], [], [])
                    for wf in sim_sphere_wireframes_list[i]:
                        wf.set_data_3d([], [], [])
                artists.append(sim_markers[i])
                artists.extend(sim_sphere_wireframes_list[i])

            # Update conflict markers
            current_conflicts_at_time = time_indexed_conflicts.get(frame_time, [])
            if current_conflicts_at_time:
                conflict_xs = [c.primary_drone_pos.x for c in current_conflicts_at_time]
                conflict_ys = [c.primary_drone_pos.y for c in current_conflicts_at_time]
                conflict_zs = [c.primary_drone_pos.z for c in current_conflicts_at_time]
                conflict_points.set_data_3d(conflict_xs, conflict_ys, conflict_zs)
            else:
                conflict_points.set_data_3d([], [], [])
            artists.append(conflict_points)

            return artists

        # Create animation
        ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=int(time_step * 1000), repeat=False)

        # Save animation
        output_filename = os.path.join(self.output_dir, f"{scenario_name}_animation.gif")
        print(f"Saving Matplotlib animation to {output_filename}...")
        try:
            ani.save(output_filename, writer='pillow', fps=int(1 / time_step) if time_step > 0 else 10)
            print(f"Matplotlib animation saved for scenario: {scenario_name}")
        except Exception as e:
            print(f"Error saving Matplotlib animation for {scenario_name}: {e}")
            print(
                "This often happens if you don't have enough RAM for the animation, or if an earlier frame failed to draw.")
            print("Try reducing `time_step` (which increases frames) or `num_lines` in _generate_sphere_points.")
        finally:
            plt.close(fig)

    def plot_scenario_plotly_animation(self,
                                       scenario_name: str,
                                       primary_mission: DroneMission,
                                       simulated_missions: List[DroneMission],
                                       conflicts: Optional[List[Conflict]],
                                       safety_buffer: float,
                                       time_step: float):
        """
        Generates an interactive 3D animated plot of drone trajectories and conflicts using Plotly.
        Saves the animation as an HTML file.
        """
        all_waypoints = self._get_all_waypoints(primary_mission, simulated_missions)

        # Determine plot limits based on all waypoints (x, y, z)
        min_x = min(wp.x for wp in all_waypoints) - (safety_buffer + 10)
        max_x = max(wp.x for wp in all_waypoints) + (safety_buffer + 10)
        min_y = min(wp.y for wp in all_waypoints) - (safety_buffer + 10)
        max_y = max(wp.y for wp in all_waypoints) + (safety_buffer + 10)
        min_z = min(wp.z for wp in all_waypoints) - (safety_buffer + 5)
        max_z = max(wp.z for wp in all_waypoints) + (safety_buffer + 5)

        # Determine the total time range for animation
        all_traj_points = self._get_all_trajectory_points(primary_mission, simulated_missions)
        if not all_traj_points:
            print("No trajectory points to animate for Plotly.")
            return

        min_time = min(wp.timestamp for wp in all_traj_points if wp.timestamp is not None)
        max_time = max(wp.timestamp for wp in all_traj_points if wp.timestamp is not None)
        effective_max_time = max(primary_mission.get_actual_mission_time_range()[1],
                                 max(sm.get_actual_mission_time_range()[1] for sm in simulated_missions if
                                     sm.get_actual_mission_time_range()[1] is not None))

        frames_times = []
        current_frame_time = min_time
        while current_frame_time <= effective_max_time + time_step:
            frames_times.append(current_frame_time)
            current_frame_time += time_step

        # Initial data for the first frame
        initial_primary_pos = primary_mission.get_position_at_time(frames_times[0]) if frames_times else None
        initial_sim_positions = [sm.get_position_at_time(frames_times[0]) for sm in
                                 simulated_missions] if frames_times else []

        initial_data = []

        # Static full trajectories - Increased width and opacity
        initial_data.append(go.Scatter3d(
            x=[wp.x for wp in primary_mission.trajectory_points],
            y=[wp.y for wp in primary_mission.trajectory_points],
            z=[wp.z for wp in primary_mission.trajectory_points],
            mode='lines',
            line=dict(color='blue', width=4, dash='dash'),
            name='Primary Trajectory (Full)',
            opacity=0.5
        ))

        # Custom color palette for simulated drones (Plotly)
        sim_color_names_plotly = [
            '#1f77b4',  # blue (default)
            '#ff7f0e',  # orange (default)
            '#2ca02c',  # green (Explicitly added for a clear green)
            '#d62728',  # red (default)
            '#9467bd',  # purple (default)
            '#8c564b',  # brown (default)
            '#e377c2',  # pink (default)
            '#7f7f7f',  # grey (default)
            '#bcbd22',  # yellow-green (default)
            '#17becf'  # cyan (default)
        ]
        # Assign colors to simulated drones by cycling through the palette
        sim_colors_for_drones = [sim_color_names_plotly[i % len(sim_color_names_plotly)] for i in
                                 range(len(simulated_missions))]

        for i, sim_mission in enumerate(simulated_missions):
            initial_data.append(go.Scatter3d(
                x=[wp.x for wp in sim_mission.trajectory_points],
                y=[wp.y for wp in sim_mission.trajectory_points],
                z=[wp.z for wp in sim_mission.trajectory_points],
                mode='lines',
                line=dict(color=sim_colors_for_drones[i], width=3, dash='dot'),
                name=f'Simulated Trajectory {sim_mission.drone_id} (Full)',
                opacity=0.4
            ))

        # Current drone positions (placeholders for animation)
        primary_x, primary_y, primary_z = (
        initial_primary_pos.x, initial_primary_pos.y, initial_primary_pos.z) if initial_primary_pos else (
        None, None, None)
        initial_data.append(go.Scatter3d(
            x=[primary_x], y=[primary_y], z=[primary_z],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name='Primary Drone',
            hoverinfo='name+x+y+z+text',
            text=f'Time: {frames_times[0]:.2f}s' if frames_times else ''
        ))

        for i, sim_pos in enumerate(initial_sim_positions):
            sim_x, sim_y, sim_z = (sim_pos.x, sim_pos.y, sim_pos.z) if sim_pos else (None, None, None)
            initial_data.append(go.Scatter3d(
                x=[sim_x], y=[sim_y], z=[sim_z],
                mode='markers',
                marker=dict(size=6, color=sim_colors_for_drones[i], symbol='circle'),  # Consistent color
                name=f'Sim Drone {simulated_missions[i].drone_id}',
                hoverinfo='name+x+y+z+text',
                text=f'Time: {frames_times[0]:.2f}s' if frames_times else ''
            ))

        # Safety buffer wireframes (placeholders for animation)
        # Primary Drone Sphere
        if initial_primary_pos:
            sphere_lines_data = self._generate_sphere_points(initial_primary_pos.x, initial_primary_pos.y,
                                                             initial_primary_pos.z, safety_buffer)
            for j, (sx, sy, sz) in enumerate(sphere_lines_data):
                initial_data.append(go.Scatter3d(
                    x=sx, y=sy, z=sz,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dot'),
                    name='Primary Safety Buffer' if j == 0 else '', showlegend=j == 0,
                    opacity=0.4
                ))

        # Simulated Drones Spheres
        for i, sim_pos in enumerate(initial_sim_positions):
            if sim_pos:
                sphere_lines_data = self._generate_sphere_points(sim_pos.x, sim_pos.y, sim_pos.z, safety_buffer)
                for j, (sx, sy, sz) in enumerate(sphere_lines_data):
                    initial_data.append(go.Scatter3d(
                        x=sx, y=sy, z=sz,
                        mode='lines',
                        line=dict(color=sim_colors_for_drones[i], width=1.5, dash='dot'),  # Consistent color
                        name=f'Sim {simulated_missions[i].drone_id} Safety Buffer' if j == 0 else '', showlegend=j == 0,
                        opacity=0.3
                    ))

        # Conflict points (placeholder for animation)
        initial_data.append(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x', line=dict(width=2, color='red')),
            name='Conflict Point',
            hoverinfo='name+x+y+z+text'
        ))

        # Create frames for animation
        plotly_frames = []
        # Cache conflicts for quick lookup during animation
        time_indexed_conflicts = {}
        if conflicts:
            for conflict in conflicts:
                conflict_time_rounded = round(conflict.time_of_conflict / time_step) * time_step
                if conflict_time_rounded not in time_indexed_conflicts:
                    time_indexed_conflicts[conflict_time_rounded] = []
                time_indexed_conflicts[conflict_time_rounded].append(conflict)

        for frame_time in frames_times:
            frame_data = []

            # Add static full trajectories to each frame - Consistent properties
            frame_data.append(go.Scatter3d(
                x=[wp.x for wp in primary_mission.trajectory_points],
                y=[wp.y for wp in primary_mission.trajectory_points],
                z=[wp.z for wp in primary_mission.trajectory_points],
                mode='lines', line=dict(color='blue', width=4, dash='dash'), opacity=0.5
            ))
            for i, sim_mission in enumerate(simulated_missions):
                frame_data.append(go.Scatter3d(
                    x=[wp.x for wp in sim_mission.trajectory_points],
                    y=[wp.y for wp in sim_mission.trajectory_points],
                    z=[wp.z for wp in sim_mission.trajectory_points],
                    mode='lines', line=dict(color=sim_colors_for_drones[i], width=3, dash='dot'), opacity=0.4
                ))

            # Current drone positions
            primary_pos = primary_mission.get_position_at_time(frame_time)
            primary_x, primary_y, primary_z = (primary_pos.x, primary_pos.y, primary_pos.z) if primary_pos else (
            None, None, None)
            frame_data.append(go.Scatter3d(
                x=[primary_x], y=[primary_y], z=[primary_z],
                mode='markers', marker=dict(size=8, color='red', symbol='circle'),
                text=f'Time: {frame_time:.2f}s'
            ))

            for i, sim_mission in enumerate(simulated_missions):
                sim_pos = sim_mission.get_position_at_time(frame_time)
                sim_x, sim_y, sim_z = (sim_pos.x, sim_pos.y, sim_pos.z) if sim_pos else (None, None, None)
                frame_data.append(go.Scatter3d(
                    x=[sim_x], y=[sim_y], z=[sim_z],
                    mode='markers', marker=dict(size=6, color=sim_colors_for_drones[i], symbol='circle'),
                    # Consistent color
                    text=f'Time: {frame_time:.2f}s'
                ))

            # Safety buffer wireframes for current frame
            if primary_pos:
                sphere_lines_data = self._generate_sphere_points(primary_pos.x, primary_pos.y, primary_pos.z,
                                                                 safety_buffer)
                for j, (sx, sy, sz) in enumerate(sphere_lines_data):
                    frame_data.append(go.Scatter3d(
                        x=sx, y=sy, z=sz, mode='lines',
                        line=dict(color='red', width=2, dash='dot'), opacity=0.4
                    ))
            for i, sim_mission in enumerate(simulated_missions):
                sim_pos = sim_mission.get_position_at_time(frame_time)
                if sim_pos:
                    sphere_lines_data = self._generate_sphere_points(sim_pos.x, sim_pos.y, sim_pos.z, safety_buffer)
                    for j, (sx, sy, sz) in enumerate(sphere_lines_data):
                        frame_data.append(go.Scatter3d(
                            x=sx, y=sy, z=sz, mode='lines',
                            line=dict(color=sim_colors_for_drones[i], width=1.5, dash='dot'), opacity=0.3
                        ))

            # Conflict points for current frame
            current_conflicts_at_time = time_indexed_conflicts.get(frame_time, [])
            if current_conflicts_at_time:
                conflict_xs = [c.primary_drone_pos.x for c in current_conflicts_at_time]
                conflict_ys = [c.primary_drone_pos.y for c in current_conflicts_at_time]
                conflict_zs = [c.primary_drone_pos.z for c in current_conflicts_at_time]
                frame_data.append(go.Scatter3d(
                    x=conflict_xs, y=conflict_ys, z=conflict_zs,
                    mode='markers', marker=dict(size=10, color='red', symbol='x', line=dict(width=2, color='red')),
                    hoverinfo='name+x+y+z+text',
                    text=[f'Time: {frame_time:.2f}s, Drone: {c.conflicting_drone_id}' for c in
                          current_conflicts_at_time]
                ))

            plotly_frames.append(go.Frame(data=frame_data, name=str(frame_time)))

        # Create the figure
        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title='X (m)', range=[min_x, max_x]),
                    yaxis=dict(title='Y (m)', range=[min_y, max_y]),
                    zaxis=dict(title='Z (m)', range=[min_z, max_z]),
                    aspectmode='data'
                ),
                title=f"Plotly Animation: Scenario: {scenario_name}",
                hovermode='closest',
                updatemenus=[{
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': int(time_step * 1000), 'redraw': True},
                                            'fromcurrent': True}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None],
                                     {'frame': {'duration': int(time_step * 1000), 'redraw': True}, 'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[f.name],
                                     {'mode': 'immediate', 'frame': {'duration': int(time_step * 1000), 'redraw': True},
                                      'transition': {'duration': 0}}],
                            'label': f'{float(f.name):.2f}s',
                            'method': 'animate'
                        } for f in plotly_frames
                    ],
                    'transition': {'duration': 0},
                    'x': 0.1,
                    'len': 0.9,
                    'currentvalue': {'font': {'size': 12}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
                    'y': 0,
                    'yanchor': 'top'
                }]
            ),
            frames=plotly_frames
        )

        output_filename = os.path.join(self.plotly_output_dir, f"{scenario_name}_plotly_animation.html")
        print(f"Saving Plotly animation to {output_filename}...")
        py_plot(fig, filename=output_filename, auto_open=False)
        print(f"Plotly animation saved for scenario: {scenario_name}")

    def plot_distance_vs_time(self,
                              scenario_name: str,
                              primary_mission: DroneMission,
                              simulated_missions: List[DroneMission],
                              conflicts: Optional[List[Conflict]],
                              safety_buffer: float,
                              plot_times: np.ndarray,  # Passed from main
                              distances_over_time: Dict[str, List[float]]  # Passed from main
                              ):
        """
        Generates a 2D plot showing the distance between the primary drone and each simulated drone over time.
        Highlights the safety buffer and conflict points.
        """
        if not plot_times.size > 0:
            print("No time points to plot distance vs. time.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Custom color palette for simulated drones (Matplotlib) - must match Plotly
        sim_color_names_mpl = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # grey
            '#bcbd22',  # yellow-green
            '#17becf'  # cyan
        ]

        for i, sim_mission in enumerate(simulated_missions):
            ax.plot(plot_times, distances_over_time[sim_mission.drone_id],
                    color=sim_color_names_mpl[i % len(sim_color_names_mpl)],
                    label=f'Dist to Drone {sim_mission.drone_id}')

        # Plot safety buffer line
        ax.axhline(y=safety_buffer, color='red', linestyle='--', linewidth=2,
                   label=f'Safety Buffer ({safety_buffer:.2f}m)')

        # Highlight conflict points (discrete points from the conflict detection logic)
        if conflicts:
            conflict_x = [c.time_of_conflict for c in conflicts]
            conflict_y = [c.distance_at_conflict for c in conflicts]
            ax.plot(conflict_x, conflict_y, 'ro', markersize=8, markeredgecolor='red',
                    markerfacecolor='none', label='Conflict Point')

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Distance (meters)")
        ax.set_title(f"Distance Between Primary Drone and Simulated Drones Over Time - Scenario: {scenario_name}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_ylim(bottom=0)

        # Save the plot
        output_filename = os.path.join(self.plots_output_dir, f"{scenario_name}_distance_vs_time.png")
        print(f"Saving Distance vs. Time plot to {output_filename}...")
        try:
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"Distance vs. Time plot saved for scenario: {scenario_name}")
        except Exception as e:
            print(f"Error saving Distance vs. Time plot for {scenario_name}: {e}")
        finally:
            plt.close(fig)

            # --- NEW: Temporal Conflict Timeline/Gantt Chart ---

    def plot_temporal_conflict_timeline(self,
                                        scenario_name: str,
                                        primary_mission_id: str,
                                        simulated_missions: List[DroneMission],
                                        safety_buffer: float,
                                        plot_times: np.ndarray,
                                        distances_over_time: Dict[str, List[float]]):
        """
        Generates a Gantt-style chart showing the temporal duration of conflicts
        between the primary drone and each simulated drone.
        """
        if not plot_times.size > 0:
            print("No time points for temporal conflict timeline.")
            return
        if not simulated_missions:
            print("No simulated missions for temporal conflict timeline.")
            return

        conflict_intervals = []
        # Identify conflict intervals from distances_over_time
        for sim_mission in simulated_missions:
            sim_drone_id = sim_mission.drone_id
            is_in_conflict = False
            conflict_start_time = None

            # Iterate through time steps and corresponding distances
            for i, current_time in enumerate(plot_times):
                distance = distances_over_time[sim_drone_id][i]

                # Check if distance is a valid number (not NaN) and below safety buffer
                if not np.isnan(distance) and distance < safety_buffer:
                    if not is_in_conflict:
                        # Conflict just started
                        is_in_conflict = True
                        conflict_start_time = current_time
                else:
                    if is_in_conflict:
                        # Conflict just ended
                        conflict_end_time = current_time
                        conflict_intervals.append({
                            'start': conflict_start_time,
                            'end': conflict_end_time,
                            'pair': f'{primary_mission_id}-{sim_drone_id}'
                        })
                        is_in_conflict = False
                        conflict_start_time = None

            # Handle case where conflict extends to the end of the simulation
            if is_in_conflict and conflict_start_time is not None:
                conflict_intervals.append({
                    'start': conflict_start_time,
                    'end': plot_times[-1],  # Conflict lasts until the end of the plotted time
                    'pair': f'{primary_mission_id}-{sim_drone_id}'
                })

        if not conflict_intervals:
            print(f"No conflict intervals detected for scenario: {scenario_name}. Not generating timeline plot.")
            return

        fig, ax = plt.subplots(figsize=(12, max(5, len(simulated_missions))))  # Adjust height based on number of drones

        # Prepare y-axis labels and positions
        unique_pairs = sorted(list(set(c['pair'] for c in conflict_intervals)))
        y_pos_map = {pair: idx for idx, pair in enumerate(unique_pairs)}
        y_labels = [f"Primary Drone ({primary_mission_id}) vs.\nSim Drone ({pair.split('-')[-1]})" for pair in
                    unique_pairs]

        # Custom color palette for Gantt bars (reuse MPL palette)
        sim_color_names_mpl = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # grey
            '#bcbd22',  # yellow-green
            '#17becf'  # cyan
        ]

        # Plot each conflict interval as a horizontal bar
        for interval in conflict_intervals:
            y_pos = y_pos_map[interval['pair']]
            duration = interval['end'] - interval['start']

            # Find the color corresponding to this simulated drone ID
            sim_drone_id = interval['pair'].split('-')[-1]
            sim_idx = next((i for i, sm in enumerate(simulated_missions) if sm.drone_id == sim_drone_id), None)
            bar_color = 'red'  # Default conflict color, but could use sim_color_names_mpl[sim_idx % len(sim_color_names_mpl)] for specific drone color.
            # Using 'red' is more indicative of a conflict.

            ax.barh(y=y_pos, width=duration, left=interval['start'], height=0.6,
                    align='center', color=bar_color, alpha=0.8)  # Red bars for conflicts

        ax.set_yticks(list(y_pos_map.values()))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Temporal Conflict Timeline - Scenario: {scenario_name}")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(plot_times[0], plot_times[-1])  # Set X-axis limits to match the simulation time
        ax.invert_yaxis()  # Often clearer for Gantt charts

        # Add a legend for the bar color (if using a fixed color for conflicts)
        if conflict_intervals:  # Only add legend if conflicts exist
            ax.legend([plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.8)], ['Conflict Duration'], loc='upper right')

        # Save the plot
        output_filename = os.path.join(self.plots_output_dir, f"{scenario_name}_conflict_timeline.png")
        print(f"Saving Temporal Conflict Timeline plot to {output_filename}...")
        try:
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"Temporal Conflict Timeline plot saved for scenario: {scenario_name}")
        except Exception as e:
            print(f"Error saving Temporal Conflict Timeline plot for {scenario_name}: {e}")
        finally:
            plt.close(fig)
