# tests/test_data_models.py
import unittest
import math
from src.models.data_models import Waypoint, DroneMission


class TestWaypoint(unittest.TestCase):
    def test_waypoint_initialization(self):
        wp = Waypoint(x=10, y=20, z=5, timestamp=100.5)
        self.assertEqual(wp.x, 10)
        self.assertEqual(wp.y, 20)
        self.assertEqual(wp.z, 5)
        self.assertEqual(wp.timestamp, 100.5)

    def test_waypoint_initialization_2d(self):
        wp = Waypoint(x=10, y=20, timestamp=100.5)  # Z defaults to 0
        self.assertEqual(wp.x, 10)
        self.assertEqual(wp.y, 20)
        self.assertEqual(wp.z, 0.0)  # Check default value
        self.assertEqual(wp.timestamp, 100.5)

    def test_distance_to(self):
        wp1 = Waypoint(0, 0, 0, 0)
        wp2 = Waypoint(3, 4, 0, 1)
        self.assertAlmostEqual(wp1.distance_to(wp2), 5.0)  # 3-4-5 triangle

        wp3 = Waypoint(0, 0, 0, 0)
        wp4 = Waypoint(1, 1, 1, 1)
        self.assertAlmostEqual(wp3.distance_to(wp4), math.sqrt(3))

    def test_to_tuple(self):
        wp = Waypoint(1, 2, 3, 4)
        self.assertEqual(wp.to_tuple(), (1, 2, 3, 4))


class TestDroneMission(unittest.TestCase):
    def test_mission_initialization(self):
        wp1 = Waypoint(0, 0, 0, 0)
        wp2 = Waypoint(10, 10, 10, 10)
        mission = DroneMission("TestDrone", [wp1, wp2], 0, 10)
        self.assertEqual(mission.drone_id, "TestDrone")
        self.assertEqual(len(mission.waypoints), 2)
        self.assertEqual(mission.mission_start_time, 0)
        self.assertEqual(mission.mission_end_time, 10)

    def test_trajectory_generation_simple(self):
        wp1 = Waypoint(0, 0, 0, 0.0)
        wp2 = Waypoint(10, 0, 0, 10.0)
        mission = DroneMission("LinearDrone", [wp1, wp2], 0.0, 10.0)
        mission.generate_interpolated_trajectory(time_step=2.0)

        # Expecting points at t=0.0, 2.0, 4.0, 6.0, 8.0, 10.0
        self.assertEqual(len(mission.trajectory_points), 6)
        self.assertAlmostEqual(mission.trajectory_points[0].x, 0.0)
        self.assertAlmostEqual(mission.trajectory_points[0].timestamp, 0.0)
        self.assertAlmostEqual(mission.trajectory_points[1].x, 2.0)
        self.assertAlmostEqual(mission.trajectory_points[1].timestamp, 2.0)
        self.assertAlmostEqual(mission.trajectory_points[-1].x, 10.0)
        self.assertAlmostEqual(mission.trajectory_points[-1].timestamp, 10.0)

    def test_trajectory_generation_multiple_segments(self):
        wp1 = Waypoint(0, 0, 0, 0.0)
        wp2 = Waypoint(10, 10, 10, 10.0)
        wp3 = Waypoint(20, 0, 0, 20.0)
        mission = DroneMission("MultiSegmentDrone", [wp1, wp2, wp3], 0.0, 20.0)
        mission.generate_interpolated_trajectory(time_step=5.0)

        # Expected: 0.0, 5.0, 10.0 (from segment 1) and 15.0, 20.0 (from segment 2)
        # 0: (0,0,0,0)
        # 1: (5,5,5,5)
        # 2: (10,10,10,10)
        # 3: (15,5,5,15)
        # 4: (20,0,0,20)
        self.assertEqual(len(mission.trajectory_points), 5)
        self.assertAlmostEqual(mission.trajectory_points[2].x, 10.0)
        self.assertAlmostEqual(mission.trajectory_points[2].y, 10.0)
        self.assertAlmostEqual(mission.trajectory_points[2].timestamp, 10.0)
        self.assertAlmostEqual(mission.trajectory_points[3].x, 15.0)
        self.assertAlmostEqual(mission.trajectory_points[3].y, 5.0)
        self.assertAlmostEqual(mission.trajectory_points[3].timestamp, 15.0)

    def test_get_position_at_time(self):
        wp1 = Waypoint(0, 0, 0, 0.0)
        wp2 = Waypoint(10, 0, 0, 10.0)
        mission = DroneMission("QueryDrone", [wp1, wp2], 0.0, 10.0)
        mission.generate_interpolated_trajectory(time_step=1.0)  # Ensure high resolution

        pos_at_5 = mission.get_position_at_time(5.0)
        self.assertIsNotNone(pos_at_5)
        self.assertAlmostEqual(pos_at_5.x, 5.0)
        self.assertAlmostEqual(pos_at_5.y, 0.0)
        self.assertAlmostEqual(pos_at_5.timestamp, 5.0)

        # Test before start time
        pos_before = mission.get_position_at_time(-1.0)
        self.assertIsNotNone(pos_before)
        self.assertAlmostEqual(pos_before.x, 0.0)  # Should return start point

        # Test after end time
        pos_after = mission.get_position_at_time(11.0)
        self.assertIsNotNone(pos_after)
        self.assertAlmostEqual(pos_after.x, 10.0)  # Should return end point

        # Test exact waypoint
        pos_at_10 = mission.get_position_at_time(10.0)
        self.assertIsNotNone(pos_at_10)
        self.assertAlmostEqual(pos_at_10.x, 10.0)
        self.assertAlmostEqual(pos_at_10.timestamp, 10.0)

    def test_get_actual_mission_time_range(self):
        wp1 = Waypoint(0, 0, 0, 10.0)
        wp2 = Waypoint(10, 10, 10, 20.0)
        mission = DroneMission("TimeRangeDrone", [wp1, wp2])
        mission.generate_interpolated_trajectory(time_step=1.0)

        start_t, end_t = mission.get_actual_mission_time_range()
        self.assertAlmostEqual(start_t, 10.0)
        self.assertAlmostEqual(end_t, 20.0)

        empty_mission = DroneMission("Empty", [])
        start_t, end_t = empty_mission.get_actual_mission_time_range()
        self.assertIsNone(start_t)
        self.assertIsNone(end_t)


if __name__ == '__main__':
    unittest.main()