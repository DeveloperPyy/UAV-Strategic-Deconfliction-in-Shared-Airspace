# tests/test_conflict_detector.py
import unittest
from src.models.data_models import Waypoint, DroneMission
from src.deconfliction.conflict_detector import check_for_conflicts, Conflict


class TestConflictDetector(unittest.TestCase):
    def setUp(self):
        # Define a common safety buffer and time step for tests
        self.safety_buffer = 5.0
        self.time_step = 1.0

    def test_no_conflict_scenario(self):
        # Primary drone: flies along X axis
        primary_waypoints = [
            Waypoint(0, 0, 0, 0.0),
            Waypoint(100, 0, 0, 100.0)
        ]
        primary_mission = DroneMission("P_Drone", primary_waypoints, 0.0, 100.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Simulated drone: flies along Y axis, far away
        sim_waypoints = [
            Waypoint(0, 50, 0, 0.0),
            Waypoint(100, 50, 0, 100.0)
        ]
        sim_mission = DroneMission("S_Drone_Safe", sim_waypoints)
        sim_mission.generate_interpolated_trajectory(self.time_step)

        status, conflicts = check_for_conflicts(
            primary_mission, [sim_mission], self.safety_buffer, self.time_step
        )
        self.assertEqual(status, "clear")
        self.assertIsNone(conflicts)

    def test_direct_collision_scenario(self):
        # Primary drone: flies from (0,0) to (10,0)
        primary_waypoints = [
            Waypoint(0, 0, 0, 0.0),
            Waypoint(10, 0, 0, 10.0)
        ]
        primary_mission = DroneMission("P_Colliding", primary_waypoints, 0.0, 10.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Simulated drone: flies from (10,0) to (0,0), directly intersecting primary
        sim_waypoints = [
            Waypoint(10, 0, 0, 0.0),
            Waypoint(0, 0, 0, 10.0)
        ]
        sim_mission = DroneMission("S_Collision", sim_waypoints)
        sim_mission.generate_interpolated_trajectory(self.time_step)

        status, conflicts = check_for_conflicts(
            primary_mission, [sim_mission], self.safety_buffer, self.time_step
        )
        self.assertEqual(status, "conflict detected")
        self.assertIsNotNone(conflicts)
        self.assertGreater(len(conflicts), 0)

        # Check a specific conflict point (e.g., at time 5.0, they should meet at (5,0,0))
        # This will depend on the exact interpolation, so check distance is below buffer
        found_conflict_at_midpoint = False
        for conflict in conflicts:
            if 4.5 <= conflict.time_of_conflict <= 5.5:  # Allow for float precision
                self.assertLess(conflict.distance_at_conflict, self.safety_buffer)
                found_conflict_at_midpoint = True
        self.assertTrue(found_conflict_at_midpoint, "Did not find expected conflict near midpoint.")

    def test_close_pass_no_conflict(self):
        # Primary: (0,0,0) to (10,0,0) in 10s
        primary_waypoints = [
            Waypoint(0, 0, 0, 0.0),
            Waypoint(10, 0, 0, 10.0)
        ]
        primary_mission = DroneMission("P_ClosePass", primary_waypoints, 0.0, 10.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Simulated: (5, 6, 0) to (5, 6, 0) in 10s (hover just outside safety buffer)
        sim_waypoints = [
            Waypoint(5, 6.0, 0, 0.0),
            Waypoint(5, 6.0, 0, 10.0)
        ]
        sim_mission = DroneMission("S_Hover_Near", sim_waypoints)
        sim_mission.generate_interpolated_trajectory(self.time_step)

        # Safety buffer is 5.0. At t=5.0, primary is at (5,0,0). Sim is at (5,6,0).
        # Distance is 6.0, which is > 5.0. So no conflict.
        status, conflicts = check_for_conflicts(
            primary_mission, [sim_mission], self.safety_buffer, self.time_step
        )
        self.assertEqual(status, "clear")
        self.assertIsNone(conflicts)

    def test_conflict_with_altitude(self):
        # Primary drone: flies along X axis at Z=10
        primary_waypoints = [
            Waypoint(0, 0, 10, 0.0),
            Waypoint(10, 0, 10, 10.0)
        ]
        primary_mission = DroneMission("P_Altitude", primary_waypoints, 0.0, 10.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Simulated drone: flies from (10,0,10) to (0,0,10)
        sim_waypoints = [
            Waypoint(10, 0, 10, 0.0),
            Waypoint(0, 0, 10, 10.0)
        ]
        sim_mission = DroneMission("S_Altitude_Conflict", sim_waypoints)
        sim_mission.generate_interpolated_trajectory(self.time_step)

        status, conflicts = check_for_conflicts(
            primary_mission, [sim_mission], self.safety_buffer, self.time_step
        )
        self.assertEqual(status, "conflict detected")
        self.assertIsNotNone(conflicts)
        self.assertGreater(len(conflicts), 0)

        # Now a non-conflict case because of altitude
        sim_waypoints_high = [
            Waypoint(10, 0, 20, 0.0),  # Z=20
            Waypoint(0, 0, 20, 10.0)  # Z=20
        ]
        sim_mission_high = DroneMission("S_Altitude_NoConflict", sim_waypoints_high)
        sim_mission_high.generate_interpolated_trajectory(self.time_step)

        status_high, conflicts_high = check_for_conflicts(
            primary_mission, [sim_mission_high], self.safety_buffer, self.time_step
        )
        # At t=5, P is (5,0,10), S_high is (5,0,20). Distance = 10.0, > 5.0 buffer
        self.assertEqual(status_high, "clear")
        self.assertIsNone(conflicts_high)

    def test_multiple_simulated_drones(self):
        primary_waypoints = [Waypoint(0, 0, 0, 0.0), Waypoint(20, 0, 0, 20.0)]
        primary_mission = DroneMission("P_Multi", primary_waypoints, 0.0, 20.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Sim drone 1: conflict around (10,0,0)
        sim1_waypoints = [Waypoint(10, 0, 0, 0.0), Waypoint(10, 0, 0, 20.0)]  # Hover at (10,0,0)
        sim1_mission = DroneMission("S1_Hovering", sim1_waypoints)
        sim1_mission.generate_interpolated_trajectory(self.time_step)

        # Sim drone 2: no conflict
        sim2_waypoints = [Waypoint(0, 50, 0, 0.0), Waypoint(20, 50, 0, 20.0)]
        sim2_mission = DroneMission("S2_Safe", sim2_waypoints)
        sim2_mission.generate_interpolated_trajectory(self.time_step)

        status, conflicts = check_for_conflicts(
            primary_mission, [sim1_mission, sim2_mission], self.safety_buffer, self.time_step
        )
        self.assertEqual(status, "conflict detected")
        self.assertIsNotNone(conflicts)
        # Expect multiple conflicts with S1_Hovering around time 10.0 (and surrounding steps)
        self.assertTrue(any(c.conflicting_drone_id == "S1_Hovering" for c in conflicts))
        self.assertFalse(any(c.conflicting_drone_id == "S2_Safe" for c in conflicts))

    def test_mission_time_window_filtering(self):
        # Primary drone: full trajectory 0-100, but mission window 20-80
        primary_waypoints = [Waypoint(0, 0, 0, 0.0), Waypoint(100, 0, 0, 100.0)]
        primary_mission = DroneMission("P_Window", primary_waypoints, 20.0, 80.0)
        primary_mission.generate_interpolated_trajectory(self.time_step)

        # Sim drone: conflicts only outside primary's mission window
        sim_waypoints = [
            Waypoint(5, 0, 0, 5.0),  # Conflict at t=5.0 (outside window)
            Waypoint(95, 0, 0, 95.0)  # Conflict at t=95.0 (outside window)
        ]
        sim_mission = DroneMission("S_OutsideWindow", sim_waypoints)
        sim_mission.generate_interpolated_trajectory(self.time_step)

        status, conflicts = check_for_conflicts(
            primary_mission, [sim_mission], self.safety_buffer, self.time_step
        )
        # Expected: No conflict because the primary drone is only checked within its defined window
        self.assertEqual(status, "clear")
        self.assertIsNone(conflicts)

        # Sim drone: conflicts inside primary's mission window
        sim_waypoints_inside = [
            Waypoint(50, 0, 0, 50.0)  # Conflict at t=50.0 (inside window)
        ]
        sim_mission_inside = DroneMission("S_InsideWindow", sim_waypoints_inside)
        sim_mission_inside.generate_interpolated_trajectory(self.time_step)

        status_inside, conflicts_inside = check_for_conflicts(
            primary_mission, [sim_mission_inside], self.safety_buffer, self.time_step
        )
        self.assertEqual(status_inside, "conflict detected")
        self.assertIsNotNone(conflicts_inside)
        self.assertTrue(any(c.time_of_conflict >= 20.0 and c.time_of_conflict <= 80.0 for c in conflicts_inside))


if __name__ == '__main__':
    unittest.main()