"""Orbital mechanics primitives — pure functions, no pygame dependency.

Units are SI-ish: pixels for distance, seconds for time. `MU = G * M_planet`
is tuned to give visually pleasant orbit periods at the game's scale.
"""

import math

MU = 3_600_000.0


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
    thrust: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Semi-implicit Euler step under gravity plus optional thrust accel."""
    ax, ay = gravity_accel(pos, planet)
    vx = vel[0] + (ax + thrust[0]) * dt
    vy = vel[1] + (ay + thrust[1]) * dt
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
