# Simulator
TOWN = "Town01"
VEHICLE_MODEL = "vehicle.mercedes.coupe"
DT = 0.05  # 20 Hz, fixed delta in synchronous mode

# Route
# AutonomousVehicle.spawn_vehicle currently hardcodes spawn_points[0] for the
# ego, so START_SPAWN_IDX must stay 0 until that is parameterized.
START_SPAWN_IDX = 0
END_SPAWN_IDX = 50
ROUTE_SAMPLING_RESOLUTION_M = 1.0

# Speed
TARGET_SPEED = 15.0  # m/s (~54 km/h)

# Lookahead
LOOKAHEAD_MIN_M = 5.0
LOOKAHEAD_TIME_S = 0.7  # lookahead = max(LOOKAHEAD_MIN_M, speed * LOOKAHEAD_TIME_S)

# PID gains (Kp, Ki, Kd)
LATERAL_PID = (1.2, 0.01, 0.3)
LONGITUDINAL_PID = (1.0, 0.05, 0.1)
INTEGRAL_LIMIT = 2.0

# Waypoint manager
WAYPOINT_SEARCH_WINDOW = 10
SLOW_DOWN_DIST_M = 15.0
SLOW_DOWN_FLOOR_SPEED = 1.0

# Termination
GOAL_REACHED_THRESHOLD_M = 5.0
