"""Orbital mechanics primitives — pure functions, no pygame dependency.

Units are SI-ish: pixels for distance, seconds for time. `MU = G * M_planet`
is tuned to give visually pleasant orbit periods at the game's scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MU = 3_600_000.0

# A gravity source: (position, gravitational parameter μ_i). Multi-source
# levels split the system mass across several point bodies — the sum of all
# μ_i for a level is the canonical MU for far-field consistency.
GravitySource = tuple[tuple[float, float], float]


def gravity_accel(
    pos: tuple[float, float],
    sources: Sequence[GravitySource],
) -> tuple[float, float]:
    """Sum of gravitational acceleration from each source on a test particle
    at `pos`. Sources at the test point are skipped (no self-singularity)."""
    ax = 0.0
    ay = 0.0
    for (sx, sy), mu in sources:
        dx = sx - pos[0]
        dy = sy - pos[1]
        r2 = dx * dx + dy * dy
        if r2 == 0.0:
            continue
        k = mu / (r2 * math.sqrt(r2))
        ax += dx * k
        ay += dy * k
    return (ax, ay)


def step(
    pos: tuple[float, float],
    vel: tuple[float, float],
    sources: Sequence[GravitySource],
    dt: float,
    thrust: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Classical RK4 step under summed gravity plus optional constant
    thrust. O(dt⁴) truncation error per step — accurate enough at
    dt=1/60 that pre-computed periodic orbits in multi-source fields
    (e.g. the level 4 figure-eight) close to within a fraction of a
    pixel per period when the same integrator is used to find them."""
    tx, ty = thrust
    px, py = pos
    vx, vy = vel

    # k1
    a1x, a1y = gravity_accel((px, py), sources)
    a1x += tx; a1y += ty
    # k2 — midpoint state under k1
    px2 = px + 0.5 * dt * vx
    py2 = py + 0.5 * dt * vy
    vx2 = vx + 0.5 * dt * a1x
    vy2 = vy + 0.5 * dt * a1y
    a2x, a2y = gravity_accel((px2, py2), sources)
    a2x += tx; a2y += ty
    # k3 — midpoint state under k2
    px3 = px + 0.5 * dt * vx2
    py3 = py + 0.5 * dt * vy2
    vx3 = vx + 0.5 * dt * a2x
    vy3 = vy + 0.5 * dt * a2y
    a3x, a3y = gravity_accel((px3, py3), sources)
    a3x += tx; a3y += ty
    # k4 — endpoint state under k3
    px4 = px + dt * vx3
    py4 = py + dt * vy3
    vx4 = vx + dt * a3x
    vy4 = vy + dt * a3y
    a4x, a4y = gravity_accel((px4, py4), sources)
    a4x += tx; a4y += ty

    new_px = px + dt / 6.0 * (vx + 2 * vx2 + 2 * vx3 + vx4)
    new_py = py + dt / 6.0 * (vy + 2 * vy2 + 2 * vy3 + vy4)
    new_vx = vx + dt / 6.0 * (a1x + 2 * a2x + 2 * a3x + a4x)
    new_vy = vy + dt / 6.0 * (a1y + 2 * a2y + 2 * a3y + a4y)
    return (new_px, new_py), (new_vx, new_vy)


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


@dataclass(frozen=True)
class Orbit:
    """Keplerian orbit around a focus. Parameterized by mean anomaly M, which
    advances linearly in time at rate `mean_motion`. Position is recovered
    via the eccentric anomaly E by Newton-solving M = E - e·sin(E). Circular
    is the special case e=0 (M = E = true anomaly). Periapsis sits at world
    angle `arg_periapsis` from the focus."""

    semi_major: float
    eccentricity: float = 0.0
    arg_periapsis: float = 0.0

    def __post_init__(self) -> None:
        if self.semi_major <= 0:
            raise ValueError(f"semi_major must be positive, got {self.semi_major}")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(f"eccentricity must be in [0, 1), got {self.eccentricity}")

    @property
    def mean_motion(self) -> float:
        return math.sqrt(MU / (self.semi_major ** 3))

    def advance(self, mean_anomaly: float, dt: float) -> float:
        return (mean_anomaly + self.mean_motion * dt) % (2.0 * math.pi)

    def _eccentric_anomaly(self, mean_anomaly: float) -> float:
        e = self.eccentricity
        if e == 0.0:
            return mean_anomaly
        # Newton on f(E) = E - e·sin(E) - M. f'(E) = 1 - e·cos(E) ≥ 1 - e > 0.
        # Converges in ≤6 iters for e ≤ 0.5; cap at 10 as a safety net.
        E = mean_anomaly
        for _ in range(10):
            f = E - e * math.sin(E) - mean_anomaly
            dE = f / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-10:
                break
        return E

    def position(
        self, mean_anomaly: float, center: tuple[float, float]
    ) -> tuple[float, float]:
        E = self._eccentric_anomaly(mean_anomaly)
        a, e, w = self.semi_major, self.eccentricity, self.arg_periapsis
        # Perifocal coords: focus at origin, periapsis on +x. b = a·√(1-e²).
        x_pf = a * (math.cos(E) - e)
        y_pf = a * math.sqrt(1.0 - e * e) * math.sin(E)
        c, s = math.cos(w), math.sin(w)
        return (
            center[0] + x_pf * c - y_pf * s,
            center[1] + x_pf * s + y_pf * c,
        )

    def velocity(self, mean_anomaly: float) -> tuple[float, float]:
        E = self._eccentric_anomaly(mean_anomaly)
        a, e, w = self.semi_major, self.eccentricity, self.arg_periapsis
        n = self.mean_motion
        # d/dt of position: dE/dt = n / (1 - e·cos E).
        denom = 1.0 - e * math.cos(E)
        vx_pf = -a * n * math.sin(E) / denom
        vy_pf = a * n * math.sqrt(1.0 - e * e) * math.cos(E) / denom
        c, s = math.cos(w), math.sin(w)
        return (vx_pf * c - vy_pf * s, vx_pf * s + vy_pf * c)

    def sample_path(
        self, center: tuple[float, float], n: int = 96
    ) -> list[tuple[float, float]]:
        """Closed loop of N points around the orbit, for guide rendering.
        Sampled uniformly in eccentric anomaly so spacing is reasonable at
        moderate eccentricities."""
        a, e, w = self.semi_major, self.eccentricity, self.arg_periapsis
        b = a * math.sqrt(1.0 - e * e)
        c, s = math.cos(w), math.sin(w)
        out: list[tuple[float, float]] = []
        for i in range(n):
            E = 2.0 * math.pi * i / n
            x_pf = a * (math.cos(E) - e)
            y_pf = b * math.sin(E)
            out.append((
                center[0] + x_pf * c - y_pf * s,
                center[1] + x_pf * s + y_pf * c,
            ))
        return out


@dataclass(frozen=True)
class Lemniscate:
    """Gerono lemniscate — parametric figure-eight curve. Phase s ∈ [0, 2π);
    period in seconds for one full s-traversal. Loop tips at (±A, 0) relative
    to center; crossover passes through the center at s = π/2 and s = 3π/2.
    Curve is rotated by `arg_rotation` from the +x axis. Decoupled from
    gravity — the stranded ship rides this fixed track regardless of the
    surrounding mass distribution."""

    semi_major: float
    period: float
    arg_rotation: float = 0.0

    def __post_init__(self) -> None:
        if self.semi_major <= 0:
            raise ValueError(f"semi_major must be positive, got {self.semi_major}")
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def angular_rate(self) -> float:
        return 2.0 * math.pi / self.period

    def advance(self, phase: float, dt: float) -> float:
        return (phase + self.angular_rate * dt) % (2.0 * math.pi)

    def _local(self, phase: float) -> tuple[float, float]:
        a = self.semi_major
        return (a * math.cos(phase), 0.5 * a * math.sin(2.0 * phase))

    def _local_velocity(self, phase: float) -> tuple[float, float]:
        a = self.semi_major
        w = self.angular_rate
        return (-a * w * math.sin(phase), a * w * math.cos(2.0 * phase))

    def position(
        self, phase: float, center: tuple[float, float]
    ) -> tuple[float, float]:
        x, y = self._local(phase)
        c, s = math.cos(self.arg_rotation), math.sin(self.arg_rotation)
        return (center[0] + x * c - y * s, center[1] + x * s + y * c)

    def velocity(self, phase: float) -> tuple[float, float]:
        vx, vy = self._local_velocity(phase)
        c, s = math.cos(self.arg_rotation), math.sin(self.arg_rotation)
        return (vx * c - vy * s, vx * s + vy * c)

    def sample_path(
        self, center: tuple[float, float], n: int = 96
    ) -> list[tuple[float, float]]:
        return [self.position(2.0 * math.pi * i / n, center) for i in range(n)]


@dataclass(frozen=True)
class TabulatedOrbit:
    """Periodic orbit stored as a closed table of (x, y, vx, vy) samples in
    body-frame coordinates (origin = system center). Phase ∈ [0, 2π) maps
    linearly to sample index; values between samples come from linear
    interpolation, with the last sample wrapping back to the first to close
    the loop. Use this for orbits that have no closed-form expression —
    e.g., a periodic orbit found by differential corrections in a
    two-fixed-center gravity field. The table itself is precomputed
    offline; this class only does cheap interpolation at runtime."""

    samples: tuple[tuple[float, float, float, float], ...]
    period: float

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise ValueError("need at least 2 samples")
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def angular_rate(self) -> float:
        return 2.0 * math.pi / self.period

    def advance(self, phase: float, dt: float) -> float:
        return (phase + self.angular_rate * dt) % (2.0 * math.pi)

    def _lookup(self, phase: float) -> tuple[float, float, float, float]:
        n = len(self.samples)
        f = (phase % (2.0 * math.pi)) * n / (2.0 * math.pi)
        i = int(f) % n
        a = f - int(f)
        s0 = self.samples[i]
        s1 = self.samples[(i + 1) % n]
        return (
            s0[0] + (s1[0] - s0[0]) * a,
            s0[1] + (s1[1] - s0[1]) * a,
            s0[2] + (s1[2] - s0[2]) * a,
            s0[3] + (s1[3] - s0[3]) * a,
        )

    def position(
        self, phase: float, center: tuple[float, float]
    ) -> tuple[float, float]:
        x, y, _, _ = self._lookup(phase)
        return (center[0] + x, center[1] + y)

    def velocity(self, phase: float) -> tuple[float, float]:
        _, _, vx, vy = self._lookup(phase)
        return (vx, vy)

    def sample_path(
        self, center: tuple[float, float], n: int = 96
    ) -> list[tuple[float, float]]:
        # If the caller asks for exactly the stored sample count we can
        # skip interpolation; otherwise re-sample at uniform phases.
        if n == len(self.samples):
            return [(center[0] + s[0], center[1] + s[1]) for s in self.samples]
        return [
            self.position(2.0 * math.pi * i / n, center) for i in range(n)
        ]
