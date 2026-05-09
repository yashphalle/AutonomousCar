import math
from contextlib import contextmanager
from dataclasses import dataclass

import carla


@dataclass
class EgoState:
    x: float
    y: float
    yaw_rad: float
    speed: float


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_ego_state(vehicle) -> EgoState:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    return EgoState(
        x=transform.location.x,
        y=transform.location.y,
        yaw_rad=math.radians(transform.rotation.yaw),
        speed=math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2),
    )


def output_to_throttle_brake(output: float) -> tuple[float, float]:
    if output >= 0:
        return min(output, 1.0), 0.0
    return 0.0, min(-output, 1.0)


def draw_route(world, route, life_time: float = 0.0) -> None:
    color = carla.Color(180, 180, 180)
    for i in range(len(route) - 1):
        a = route[i][0].transform.location
        b = route[i + 1][0].transform.location
        world.debug.draw_line(
            carla.Location(x=a.x, y=a.y, z=a.z + 0.3),
            carla.Location(x=b.x, y=b.y, z=b.z + 0.3),
            thickness=0.08,
            color=color,
            life_time=life_time,
        )


def draw_next_waypoints(
    world,
    route,
    current_idx: int,
    n: int = 50,
    life_time: float = 1,
) -> None:
    end_idx = min(current_idx + n, len(route))
    for offset, idx in enumerate(range(current_idx, end_idx)):
        loc = route[idx][0].transform.location
        if offset == 0:
            color = carla.Color(0, 255, 0)
            size = 0.18
        else:
            color = carla.Color(255, 0, 0)
            size = 0.06
        world.debug.draw_point(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5),
            size=size,
            color=color,
            life_time=life_time,
        )


@contextmanager
def synchronous_mode(world, fixed_dt: float = 0.05):
    settings = world.get_settings()
    original_sync = settings.synchronous_mode
    original_dt = settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(settings)
    try:
        yield
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = original_sync
        settings.fixed_delta_seconds = original_dt
        world.apply_settings(settings)
