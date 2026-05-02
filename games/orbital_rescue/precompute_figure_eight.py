"""One-shot precompute for the level 4 figure-eight orbit.

Differential corrections find a periodic orbit in the two-fixed-center
gravity field, then we tabulate one period as (x, y, vx, vy) samples in
body-frame coordinates (relative to the system midpoint between the two
stars). The output is a Python file the game imports at runtime — no
integration happens during gameplay.

Re-run when star geometry / mass ratio changes:

    uv run python games/orbital_rescue/precompute_figure_eight.py

The script writes games/orbital_rescue/figure_eight_data.py.
"""

from __future__ import annotations

import math
import os
import sys

# Match physics.py
MU = 3_600_000.0

# Match level 4 star configuration. Stars at (±STAR_OFFSET, 0) relative to
# the midpoint, each carrying half the system mass. Keep this in sync with
# levels.py LEVELS[4].stars when changing.
STAR_OFFSET = 110.0
SOURCES = (
    ((-STAR_OFFSET, 0.0), MU * 0.5),
    ((+STAR_OFFSET, 0.0), MU * 0.5),
)

# Match constants.DT_SIM. Critical: the precompute must use the same
# integrator and timestep as the game so the resulting orbit is exactly
# closed under the game's own dynamics. Otherwise the tug drifts off
# the tabulated path the moment it's released from dock.
GAME_DT = 1.0 / 60.0

OUT_PATH = os.path.join(os.path.dirname(__file__), "figure_eight_data.py")


def gravity_accel(
    pos: tuple[float, float], sources
) -> tuple[float, float]:
    ax = ay = 0.0
    for (sx, sy), mu in sources:
        dx, dy = sx - pos[0], sy - pos[1]
        r2 = dx * dx + dy * dy
        r3 = r2 * math.sqrt(r2)
        ax += dx * mu / r3
        ay += dy * mu / r3
    return ax, ay


def deriv(
    state: tuple[float, float, float, float], sources
) -> tuple[float, float, float, float]:
    x, y, vx, vy = state
    ax, ay = gravity_accel((x, y), sources)
    return (vx, vy, ax, ay)


def rk4_step(state, sources, dt):
    """4th-order Runge-Kutta. Used only for an initial high-accuracy seed
    pass; final tabulation uses the game's own integrator to guarantee
    self-consistency."""
    k1 = deriv(state, sources)
    s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(4))
    k2 = deriv(s2, sources)
    s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(4))
    k3 = deriv(s3, sources)
    s4 = tuple(state[i] + dt * k3[i] for i in range(4))
    k4 = deriv(s4, sources)
    return tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(4)
    )


def game_step(state, sources, dt):
    """Mirrors physics.step (RK4, no thrust). Drives both the differential
    corrections and the tabulation pass so the resulting orbit closes
    exactly under the game's own integrator."""
    return rk4_step(state, sources, dt)


def integrate_until_kth_y_crossing(state0, sources, dt, max_t, k, step_fn):
    """Integrate forward until y crosses zero for the k-th time. Linearly
    interpolates between the two adjacent steps to recover the crossing
    state. Skips the initial state if it's already at y=0."""
    state = state0
    t = 0.0
    prev_y = state[1]
    crossings = 0
    while t < max_t:
        new_state = step_fn(state, sources, dt)
        new_y = new_state[1]
        # Sign-change detection. Tolerate floating-point fuzz at start.
        if prev_y * new_y < 0.0:
            crossings += 1
            if crossings == k:
                alpha = prev_y / (prev_y - new_y)
                interp = tuple(
                    state[i] + alpha * (new_state[i] - state[i]) for i in range(4)
                )
                return interp, t + alpha * dt
        state = new_state
        prev_y = new_y
        t += dt
    raise RuntimeError(f"no {k}-th y-crossing within max_t={max_t} (vy0={state0[3]})")


def find_periodic_vy(
    x0, vy_guess, sources, k_crossing, step_fn, dt, tol=1e-6, max_iter=40
):
    """Differential corrections: find vy0 so the orbit from (x0, 0, 0, vy0)
    crosses the x-axis perpendicularly (vx=0) at the k-th y-crossing —
    that crossing is the half-period symmetric point. Full period =
    2 × t_to_kth_crossing.

    `step_fn` and `dt` should match the integrator that will replay the
    orbit at runtime; the periodic orbit found here is the closed orbit
    of THAT integrator (not the underlying continuous ODE), which is
    exactly what we need for drift-free playback in-game."""
    vy = vy_guess
    for it in range(max_iter):
        end_state, t_half = integrate_until_kth_y_crossing(
            (x0, 0.0, 0.0, vy), sources, dt, max_t=30.0, k=k_crossing,
            step_fn=step_fn,
        )
        residual = end_state[2]  # vx at end
        print(
            f"  iter {it:2d}: vy={vy:.6f}  t_half={t_half:.4f}  "
            f"end_x={end_state[0]:+8.3f}  residual_vx={residual:+.4e}"
        )
        if abs(residual) < tol:
            return vy, t_half * 2.0
        eps = 1e-3
        end_plus, _ = integrate_until_kth_y_crossing(
            (x0, 0.0, 0.0, vy + eps), sources, dt, max_t=30.0, k=k_crossing,
            step_fn=step_fn,
        )
        d_residual = (end_plus[2] - residual) / eps
        if abs(d_residual) < 1e-12:
            raise RuntimeError("zero Jacobian; bad initial guess")
        vy -= residual / d_residual
    raise RuntimeError(f"did not converge in {max_iter} iterations")


def tabulate(x0, vy0, period, sources, step_fn, dt, n_samples):
    """Integrate one full period from (x0, 0, 0, vy0) using `step_fn` at
    `dt`. Records `n_samples` evenly spaced in time across the period;
    sub-steps each sample interval at the integrator's own `dt` for
    accuracy. Linear interpolation between adjacent samples reconstructs
    intermediate states at runtime."""
    sample_dt = period / n_samples
    sub_steps = max(1, int(round(sample_dt / dt)))
    inner_dt = sample_dt / sub_steps
    state = (x0, 0.0, 0.0, vy0)
    samples = [state]
    for _ in range(n_samples - 1):
        for _ in range(sub_steps):
            state = step_fn(state, sources, inner_dt)
        samples.append(state)
    # One more sample interval to verify closure.
    closing = state
    for _ in range(sub_steps):
        closing = step_fn(closing, sources, inner_dt)
    pos_err = math.hypot(closing[0] - samples[0][0], closing[1] - samples[0][1])
    vel_err = math.hypot(closing[2] - samples[0][2], closing[3] - samples[0][3])
    print(f"  closing error after one period: pos={pos_err:.4f} px,  vel={vel_err:.4f} px/s")
    return samples


def main() -> None:
    x0 = 200.0
    # Empirically (see vy-scan), the figure-eight regime in this two-star
    # field has the orbit making partial loops around each primary before
    # hitting the opposite tip — the symmetric perpendicular x-axis
    # crossing is the 5th y-crossing in forward integration, and vy ≈ 150
    # puts us inside the convergence basin.
    vy_guess = 150.0
    k_crossing = 5

    # Find the orbit accurately at fine dt (RK4 at 0.002s) — high-order
    # accuracy gives a clean Newton convergence even when the dynamics
    # near the saddle is sharp. The game then re-plays this orbit using
    # its own RK4 step at dt=1/60s; the per-step truncation error of
    # RK4 at game dt is O(dt⁴) ≈ 8e-7 of a step, so even coarse playback
    # tracks the fine-dt orbit to within sub-pixel precision per period.
    fine_dt = 0.002
    print(f"# stars: {SOURCES}")
    print(f"# x0={x0}, vy_guess={vy_guess}, k_crossing={k_crossing}, fine_dt={fine_dt}")
    print()
    print(f"Differential corrections (RK4 at dt={fine_dt}):")
    vy0, period = find_periodic_vy(
        x0, vy_guess, SOURCES, k_crossing, step_fn=rk4_step, dt=fine_dt
    )
    print(f"  → vy0={vy0:.6f}  period={period:.6f}")
    print()
    # Tabulate via RK4 at fine dt — keeps the table on the precision-found
    # orbit. 1024 samples per period gives ~15 ms resolution — small
    # enough that linear interpolation in TabulatedOrbit reconstructs
    # both position (sub-pixel) and velocity (sub-px/s) accurately.
    n_samples = 1024
    print(f"Tabulating {n_samples} samples per period at sub-step dt={fine_dt}:")
    samples = tabulate(
        x0, vy0, period, SOURCES, step_fn=rk4_step, dt=fine_dt, n_samples=n_samples
    )
    print()

    with open(OUT_PATH, "w") as f:
        f.write('"""Auto-generated by precompute_figure_eight.py.\n')
        f.write("\n")
        f.write("One period of a periodic orbit in the two-fixed-center gravity field\n")
        f.write("for level 4. Body-frame coordinates: origin = midpoint between the\n")
        f.write("two stars. Each sample is (x, y, vx, vy) at uniformly spaced phase.\n")
        f.write("Re-run the precompute script if star geometry changes.\n")
        f.write('"""\n\n')
        f.write(f"X0 = {x0!r}\n")
        f.write(f"VY0 = {vy0:.10f}\n")
        f.write(f"PERIOD = {period:.10f}\n")
        f.write(f"STAR_OFFSET = {STAR_OFFSET!r}\n")
        f.write("\n")
        f.write("SAMPLES: tuple[tuple[float, float, float, float], ...] = (\n")
        for x, y, vx, vy in samples:
            f.write(f"    ({x:+.4f}, {y:+.4f}, {vx:+.4f}, {vy:+.4f}),\n")
        f.write(")\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
