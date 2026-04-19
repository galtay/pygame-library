"""Orbital mechanics primitives — pure functions, no pygame dependency.

Units are whatever the caller chooses; typical use is pixels for distance
and frames for time. `MU = G * M_planet` is tuned (see `main.py`) to give
visually pleasant orbit periods at 60 fps.
"""

import math

MU = 1000.0


def gravity_accel(
    pos: tuple[float, float],
    planet: tuple[float, float],
) -> tuple[float, float]:
    dx = planet[0] - pos[0]
    dy = planet[1] - pos[1]
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    if r == 0.0:
        return (0.0, 0.0)
    k = MU / (r2 * r)
    return (dx * k, dy * k)


def step(
    pos: tuple[float, float],
    vel: tuple[float, float],
    planet: tuple[float, float],
    dt: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Semi-implicit Euler step under gravity from a single point mass."""
    ax, ay = gravity_accel(pos, planet)
    vx = vel[0] + ax * dt
    vy = vel[1] + ay * dt
    px = pos[0] + vx * dt
    py = pos[1] + vy * dt
    return (px, py), (vx, vy)


def circular_orbit_speed(radius: float) -> float:
    return math.sqrt(MU / radius)


def circular_orbit_angular_speed(radius: float) -> float:
    return math.sqrt(MU / (radius * radius * radius))


def circular_orbit_position(
    center: tuple[float, float],
    radius: float,
    angle: float,
) -> tuple[float, float]:
    return (
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    )


def circular_orbit_velocity(
    radius: float,
    angle: float,
) -> tuple[float, float]:
    """Tangential velocity of a prograde circular orbit at (radius, angle)."""
    speed = circular_orbit_speed(radius)
    return (-speed * math.sin(angle), speed * math.cos(angle))
