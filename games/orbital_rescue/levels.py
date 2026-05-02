"""Level definitions for Orbital Rescue.

Each level fixes the stranded ship's path, the gravitational sources the
tug experiences, the starting phase, and per-level toggles like tug launch
direction and solar-flare instability. Level 1 preserves the original
circular setup; subsequent levels vary path shape, mass distribution, and
hazard config without re-tuning damage rates or capture parameters (those
stay on the difficulty preset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import constants
import figure_eight_data
from physics import MU, GravitySource, Lemniscate, Orbit, TabulatedOrbit

# A path the stranded ship rides — Keplerian orbit, parametric curve, or
# precomputed periodic orbit. All expose position(phase, center),
# velocity(phase), advance(phase, dt), sample_path(center, n).
StrandedPath = Union[Orbit, Lemniscate, TabulatedOrbit]


def _single_star() -> tuple[GravitySource, ...]:
    """Default gravity field for single-star levels — STAR_POS at full MU."""
    return (((float(constants.STAR_POS[0]), float(constants.STAR_POS[1])), MU),)


@dataclass(frozen=True)
class Level:
    name: str
    stranded_orbit: StrandedPath
    start_mean_anomaly: float
    # Gravity sources active during the mission. Default is a single star at
    # STAR_POS; multi-star levels split MU across multiple bodies.
    stars: tuple[GravitySource, ...] = field(default_factory=_single_star)
    # When True, the tug launches with its initial circular velocity reversed
    # — it orbits the star in the opposite sense from the stranded vessel.
    # Capture requires the player to thrust against their own initial motion.
    tug_retrograde: bool = False
    # When False, the star instability/flare system is disabled for this
    # level — no countdown, no radiation zone, no per-ship damage accrual.
    flares: bool = True


_STAR_X, _STAR_Y = float(constants.STAR_POS[0]), float(constants.STAR_POS[1])

LEVELS: dict[int, Level] = {
    1: Level(
        name="circular",
        stranded_orbit=Orbit(semi_major=220.0, eccentricity=0.0, arg_periapsis=0.0),
        start_mean_anomaly=math.pi,
    ),
    2: Level(
        name="elliptical",
        stranded_orbit=Orbit(semi_major=220.0, eccentricity=0.35, arg_periapsis=0.0),
        start_mean_anomaly=math.pi,
    ),
    3: Level(
        name="retrograde",
        stranded_orbit=Orbit(semi_major=220.0, eccentricity=0.0, arg_periapsis=0.0),
        start_mean_anomaly=math.pi,
        tug_retrograde=True,
    ),
    4: Level(
        name="figure-eight",
        # Two stars at (±110, 0) from STAR_POS, half mass each → combined
        # far-field strength matches a single MU star. The stranded ship
        # rides a real periodic orbit in this two-fixed-center field
        # (precomputed offline by precompute_figure_eight.py via
        # differential corrections). Because the orbit is gravitationally
        # consistent, releasing the tug at any phase keeps it on the same
        # closed path — same "stable orbit after dock" feel as levels 1–3.
        stars=(
            ((_STAR_X - figure_eight_data.STAR_OFFSET, _STAR_Y), MU * 0.5),
            ((_STAR_X + figure_eight_data.STAR_OFFSET, _STAR_Y), MU * 0.5),
        ),
        stranded_orbit=TabulatedOrbit(
            samples=figure_eight_data.SAMPLES,
            period=figure_eight_data.PERIOD,
        ),
        start_mean_anomaly=0.0,  # phase 0 = (X0, 0) right loop tip
        flares=False,
    ),
}

LEVEL_KEYS: tuple[int, ...] = tuple(sorted(LEVELS))

LEVEL_KEYS: tuple[int, ...] = tuple(sorted(LEVELS))
