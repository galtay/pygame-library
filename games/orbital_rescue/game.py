"""Orbital Rescue — pilot a tug from the rescue ship to a stranded ship in
circular orbit around a star, then pilot the combined craft home.

Three phases:

1. OUTBOUND  — tug starts in a circular orbit at the rescue ship's radius.
               Player rotates + thrusts to intercept the stranded ship.
2. DOCKED    — combined body rides the stranded ship's circular orbit.
               Player chooses when to engage the tug controls.
3. HOMEBOUND — player pilots the tug + cargo back to the rescue ship's
               fixed position, fighting gravity.

The loop runs a fixed-timestep simulation driven by real elapsed time, so
gameplay is identical regardless of the display refresh rate. Rendering
runs once per loop iteration at whatever rate the platform delivers. The
loop is async so the same code runs under pygbag in the browser; the
`await asyncio.sleep(0)` after each flip yields to the JS event loop.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import pygame

import constants
from geometry import tug_visual_center
from levels import LEVELS, LEVEL_KEYS
from physics import (
    circular_orbit_velocity,
    step,
)
import render


class State(Enum):
    ARRIVING = auto()
    PARKED = auto()
    OUTBOUND = auto()
    DOCKED = auto()
    HOMEBOUND = auto()
    RETURNING = auto()
    WON = auto()
    FAILED = auto()


@dataclass
class InputFrame:
    left: bool
    right: bool
    thrust: bool


def tug_launch_orbit(
    retrograde: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Position, velocity, and facing of the tug on the rescue ship's circular orbit.

    Facing matches the orientation the tug had while nested inside the rescue
    ship — pointing outward from the star — so launching produces no snap.
    `retrograde=True` flips the circular velocity vector for levels where the
    tug starts orbiting against the stranded ship's motion.
    """
    pos = (float(constants.RESCUE_POS[0]), float(constants.RESCUE_POS[1]))
    vx, vy = circular_orbit_velocity(constants.RESCUE_ORBIT_RADIUS, 0.0)
    if retrograde:
        vx, vy = -vx, -vy
    facing = math.atan2(constants.RESCUE_POS[1] - constants.STAR_POS[1], constants.RESCUE_POS[0] - constants.STAR_POS[0])
    return pos, (vx, vy), facing


BRIEFING_TITLE = "MISSION BRIEFING"
# Each entry is one paragraph as a segmented stream — visual line breaks are
# applied at render time via wrap_segments. Empty entries are paragraph gaps.
BRIEFING_LINES: list[list[tuple[str, tuple[int, int, int]]]] = [
    [
        ("A ", constants.HUD),
        ("stranded", constants.STRANDED),
        (" vessel drifts in low orbit around an unstable star. When it flares, solar energy will fill the inner zone.", constants.HUD),
    ],
    [],
    [
        ("Launch the hardened ", constants.HUD),
        ("tug", constants.TUG),
        (", dock with the ", constants.HUD),
        ("stranded", constants.STRANDED),
        (" vessel, and haul it back before the flares overwhelm them.", constants.HUD),
    ],
]

PILOTING_CONTROLS: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("[LEFT/RIGHT]", constants.HUD_ACTION), (" rotate", constants.HUD)],
    [("[UP]", constants.HUD_ACTION), (" thrust", constants.HUD)],
]
SESSION_CONTROLS: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("[R]", constants.HUD_ACTION), (" reset", constants.HUD)],
    [("[ESC]", constants.HUD_ACTION), (" quit", constants.HUD)],
]


def rescue_ship_position(arrival_progress: float) -> tuple[float, float]:
    """Rescue ship's current spot: radial descent from the start radius to the station position with smoothstep easing."""
    if arrival_progress >= 1.0:
        return (float(constants.RESCUE_POS[0]), float(constants.RESCUE_POS[1]))
    t = max(0.0, arrival_progress)
    s = t * t * (3 - 2 * t)
    radius = constants.ARRIVAL_START_RADIUS + (constants.RESCUE_ORBIT_RADIUS - constants.ARRIVAL_START_RADIUS) * s
    return (float(constants.STAR_POS[0] + radius), float(constants.STAR_POS[1]))


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Shortest-arc angle interpolation from `a` to `b` by fraction `t`."""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t


def _relative_speed(
    tug_vel: tuple[float, float], target_vel: tuple[float, float]
) -> float:
    """Magnitude of the tug-vs-target relative velocity. Direction-agnostic.
    At the moment of capture this is also the impact speed."""
    return math.hypot(tug_vel[0] - target_vel[0], tug_vel[1] - target_vel[1])


@dataclass
class GameState:
    state: State
    tug_pos: tuple[float, float]
    tug_vel: tuple[float, float]
    tug_angle: float
    tug_trail: deque[tuple[float, float]]
    # Stranded ship's orbit phase as mean anomaly M (radians). Position and
    # velocity are derived from M via the level's Orbit each tick.
    stranded_anomaly: float
    stranded_pos: tuple[float, float]
    # Active level — picks the stranded ship's orbit from LEVELS.
    level: int
    arrival_progress: float
    # Shared timer for the two game-driven animations: the DOCKED rotate-to-
    # prograde and the RETURNING glide-into-rescue. They never overlap.
    phase_anim_elapsed: float
    dock_tug_start_facing: float
    # Tug's position at capture, expressed as an offset from the stranded
    # ship's cargo-mount point at that instant. The DOCKED animation rides
    # along the live stranded orbit and decays this offset to (0, 0) over
    # PHASE_ANIM_DURATION — works for any orbit shape (circular, elliptical,
    # lemniscate, …) and any number of gravity sources.
    dock_tug_offset_at_capture: tuple[float, float]
    dock_tug_start_pos: tuple[float, float]
    dock_stranded_start_pos: tuple[float, float]
    rv_stranded_at_lock: float
    rv_rescue_at_lock: float
    rv_rescue_locked: bool
    # Total time from launch_outbound() to WON. Ticks during OUTBOUND,
    # DOCKED, HOMEBOUND, and RETURNING — i.e., everything after the player
    # commits to the mission, including the game-driven dock/return anims.
    mission_elapsed: float
    # Camera zoom (1.0 inside INNER_RADIUS, eases toward MIN_ZOOM as the tug
    # approaches LOST_RADIUS). Always star-anchored.
    view_zoom: float
    # Fuel remaining in the tug, in FUEL_CAPACITY units. Only thrust burns it.
    fuel: float
    # Cumulative radiation damage. tug_damage ticks only inside the danger
    # zone; stranded_damage ticks continuously during OUTBOUND and inside
    # the danger zone during HOMEBOUND. Both reset only on launch_outbound.
    tug_damage: float
    stranded_damage: float
    has_cargo: bool
    status: str

    @classmethod
    def new(cls, *, level: int = 1, with_arrival: bool = False) -> "GameState":
        """Fresh game on `level`: tug parked on the rescue ship, stranded
        ship placed at the level's fixed start phase. Start phase is the
        same across levels so timing/score comparisons are meaningful.

        `with_arrival=True` runs the intro animation; reset (R) uses the
        default so the player jumps straight to the settled state.
        """
        level_def = LEVELS[level]
        pos, _vel, facing = tug_launch_orbit(retrograde=level_def.tug_retrograde)
        stranded_anomaly = level_def.start_mean_anomaly
        return cls(
            state=State.ARRIVING if with_arrival else State.PARKED,
            tug_pos=pos,
            tug_vel=(0.0, 0.0),
            tug_angle=facing,
            tug_trail=deque(maxlen=constants.MAX_TRAIL_POINTS),
            stranded_anomaly=stranded_anomaly,
            stranded_pos=level_def.stranded_orbit.position(
                stranded_anomaly, constants.STAR_POS
            ),
            level=level,
            arrival_progress=0.0 if with_arrival else 1.0,
            phase_anim_elapsed=0.0,
            dock_tug_start_facing=0.0,
            dock_tug_offset_at_capture=(0.0, 0.0),
            dock_tug_start_pos=(0.0, 0.0),
            dock_stranded_start_pos=(0.0, 0.0),
            rv_stranded_at_lock=0.0,
            rv_rescue_at_lock=0.0,
            rv_rescue_locked=False,
            mission_elapsed=0.0,
            view_zoom=1.0,
            fuel=constants.FUEL_CAPACITY,
            tug_damage=0.0,
            stranded_damage=0.0,
            has_cargo=False,
            status=(
                "Rescue ship arriving..." if with_arrival
                else "SPACE to launch the tug."
            ),
        )

    def finish_arrival(self) -> None:
        self.arrival_progress = 1.0
        self.state = State.PARKED
        self.status = "SPACE to launch the tug."

    def launch_outbound(self) -> None:
        pos, vel, facing = tug_launch_orbit(
            retrograde=LEVELS[self.level].tug_retrograde
        )
        self.tug_pos = pos
        self.tug_vel = vel
        self.tug_angle = facing
        self.tug_trail.clear()
        self.tug_trail.append(pos)
        self.mission_elapsed = 0.0
        self.fuel = constants.FUEL_CAPACITY
        self.tug_damage = 0.0
        self.stranded_damage = 0.0
        self.state = State.OUTBOUND
        self.status = "Pilot the tug to the stranded ship."

    def engage_homebound(self) -> None:
        vel = LEVELS[self.level].stranded_orbit.velocity(self.stranded_anomaly)
        self.tug_pos = self.stranded_pos
        self.tug_vel = vel
        # Preserve the post-alignment orientation so HOMEBOUND starts with no snap.
        self.tug_angle = constants.STRANDED_FACING
        self.tug_trail.clear()
        self.tug_trail.append(self.tug_pos)
        self.state = State.HOMEBOUND
        self.status = "Piloting home with cargo."


def simulate(
    gs: GameState,
    dt: float,
    inp: InputFrame,
    capture_radius: float,
    damage_mult: float,
    dock_speed_limit: float,
) -> None:
    """Advance the game world by one fixed simulation step (`dt` seconds)."""
    if gs.state in (State.OUTBOUND, State.DOCKED, State.HOMEBOUND, State.RETURNING):
        gs.mission_elapsed += dt
    _update_view_zoom(gs, dt)
    if gs.state is State.ARRIVING:
        _tick_arriving(gs, dt)
    elif gs.state is State.PARKED:
        _advance_stranded(gs, dt)
    elif gs.state is State.OUTBOUND:
        _tick_outbound(gs, dt, inp, capture_radius, damage_mult, dock_speed_limit)
    elif gs.state is State.DOCKED:
        _tick_docked(gs, dt, damage_mult)
    elif gs.state is State.HOMEBOUND:
        _tick_homebound(gs, dt, inp, capture_radius, damage_mult, dock_speed_limit)
    elif gs.state is State.RETURNING:
        _tick_returning(gs, dt)
    # WON, FAILED — no-op


def _tick_arriving(gs: GameState, dt: float) -> None:
    _advance_stranded(gs, dt)
    gs.arrival_progress += dt / constants.ARRIVAL_DURATION
    if gs.arrival_progress >= 1.0:
        gs.finish_arrival()


def _tick_outbound(
    gs: GameState,
    dt: float,
    inp: InputFrame,
    capture_radius: float,
    damage_mult: float,
    dock_speed_limit: float,
) -> None:
    _advance_stranded(gs, dt)
    _apply_pilot_input(gs, dt, inp)
    if _hit_star(gs):
        _fail(gs, "Crashed the tug into the star.")
        return
    if _flares_active(gs):
        tug_maxed, stranded_maxed = _accumulate_damage(gs, dt, damage_mult)
        if tug_maxed:
            _fail(gs, "Tug destroyed by solar flares.")
            return
        if stranded_maxed:
            _fail(gs, "Stranded vessel lost to solar flares.")
            return
    if _captured_stranded(gs, capture_radius, dock_speed_limit):
        return
    if _piloting_lost(gs):
        _fail(gs, "Tug lost in space.")


def _tick_docked(
    gs: GameState, dt: float, damage_mult: float
) -> None:
    _advance_stranded(gs, dt)
    gs.phase_anim_elapsed += dt
    if _flares_active(gs):
        tug_maxed, stranded_maxed = _accumulate_damage(gs, dt, damage_mult)
        if tug_maxed:
            _fail(gs, "Tug destroyed by solar flares.")
            return
        if stranded_maxed:
            _fail(gs, "Stranded vessel lost to solar flares.")
            return


def _tick_homebound(
    gs: GameState,
    dt: float,
    inp: InputFrame,
    capture_radius: float,
    damage_mult: float,
    dock_speed_limit: float,
) -> None:
    _apply_pilot_input(gs, dt, inp)
    gs.stranded_pos = gs.tug_pos
    if _hit_star(gs):
        _fail(gs, "Crashed on the way home.")
        return
    if _flares_active(gs):
        tug_maxed, stranded_maxed = _accumulate_damage(gs, dt, damage_mult)
        if tug_maxed:
            _fail(gs, "Tug destroyed by solar flares.")
            return
        if stranded_maxed:
            _fail(gs, "Stranded vessel lost to solar flares.")
            return
    if _captured_rescue(gs, capture_radius, dock_speed_limit):
        return
    if _piloting_lost(gs):
        _fail(gs, "Lost on the way home.")


def _tick_returning(gs: GameState, dt: float) -> None:
    gs.phase_anim_elapsed += dt
    if gs.phase_anim_elapsed >= constants.PHASE_ANIM_DURATION:
        gs.tug_pos = (float(constants.RESCUE_POS[0]), float(constants.RESCUE_POS[1]))
        gs.stranded_pos = gs.tug_pos
        gs.tug_angle = 0.0
        gs.state = State.WON
        gs.status = "Rescue complete!"


def _apply_pilot_input(gs: GameState, dt: float, inp: InputFrame) -> None:
    if inp.left:
        gs.tug_angle -= constants.TUG_ROT_SPEED * dt
    if inp.right:
        gs.tug_angle += constants.TUG_ROT_SPEED * dt
    if inp.thrust and gs.fuel > 0.0:
        thrust = (
            math.cos(gs.tug_angle) * constants.TUG_THRUST,
            math.sin(gs.tug_angle) * constants.TUG_THRUST,
        )
        gs.fuel = max(0.0, gs.fuel - constants.FUEL_BURN_RATE * dt)
    else:
        thrust = (0.0, 0.0)
    sources = LEVELS[gs.level].stars
    gs.tug_pos, gs.tug_vel = step(gs.tug_pos, gs.tug_vel, sources, dt, thrust)
    gs.tug_trail.append(gs.tug_pos)


def _hit_star(gs: GameState) -> bool:
    """True iff the tug overlaps any star core in the active level."""
    r2 = constants.STAR_RADIUS * constants.STAR_RADIUS
    for (sx, sy), _mu in LEVELS[gs.level].stars:
        dx = gs.tug_pos[0] - sx
        dy = gs.tug_pos[1] - sy
        if dx * dx + dy * dy <= r2:
            return True
    return False


def _flares_active(gs: GameState) -> bool:
    """True iff stellar flares are currently active (post-instability-delay)
    on a level that uses flares."""
    return (
        LEVELS[gs.level].flares
        and gs.mission_elapsed >= constants.INSTABILITY_DELAY
    )


def _flare_growth(gs: GameState) -> float:
    """0.0 at launch, 1.0 once the instability countdown reaches 0. Drives
    the staggered buildup of flare layers in the renderer — inner rings
    appear first, outer last, so the danger zone has fully formed by the
    moment damage begins. Damage itself remains binary on/off. Stays at 0
    on levels with `flares=False` so the radiation zone is never drawn."""
    if not LEVELS[gs.level].flares:
        return 0.0
    return min(1.0, gs.mission_elapsed / constants.INSTABILITY_DELAY)


def _accumulate_damage(gs: GameState, dt: float, damage_mult: float) -> tuple[bool, bool]:
    """Tick radiation damage on both ships using their own positions and a
    single danger radius. Both rates are scaled by the difficulty's
    damage_mult; the hardened tug already takes damage 3x slower than the
    stranded vessel at baseline. Returns (tug_maxed, stranded_maxed)."""
    danger_sq = constants.DAMAGE_DANGER_RADIUS * constants.DAMAGE_DANGER_RADIUS
    sx, sy = constants.STAR_POS
    tdx, tdy = gs.tug_pos[0] - sx, gs.tug_pos[1] - sy
    if tdx * tdx + tdy * tdy < danger_sq:
        gs.tug_damage = min(
            constants.DAMAGE_CAPACITY,
            gs.tug_damage + constants.TUG_DAMAGE_RATE * damage_mult * dt,
        )
    sdx, sdy = gs.stranded_pos[0] - sx, gs.stranded_pos[1] - sy
    if sdx * sdx + sdy * sdy < danger_sq:
        gs.stranded_damage = min(
            constants.DAMAGE_CAPACITY,
            gs.stranded_damage + constants.STRANDED_DAMAGE_RATE * damage_mult * dt,
        )
    return (
        gs.tug_damage >= constants.DAMAGE_CAPACITY,
        gs.stranded_damage >= constants.DAMAGE_CAPACITY,
    )


def _piloting_lost(gs: GameState) -> bool:
    dx = gs.tug_pos[0] - constants.STAR_POS[0]
    dy = gs.tug_pos[1] - constants.STAR_POS[1]
    return dx * dx + dy * dy > constants.LOST_RADIUS * constants.LOST_RADIUS


def _update_view_zoom(gs: GameState, dt: float) -> None:
    """Ease the camera toward a target zoom each tick.

    Target = 1.0 except while piloting outside INNER_RADIUS, where it shrinks
    so the tug stays exactly at the edge of the visible frame. WON and
    FAILED freeze the zoom so the modal renders over the moment-of-truth
    framing (e.g. the player sees how far they drifted on lost-in-space).
    """
    if gs.state in (State.WON, State.FAILED):
        return
    if gs.state in (State.OUTBOUND, State.HOMEBOUND):
        dx = gs.tug_pos[0] - constants.STAR_POS[0]
        dy = gs.tug_pos[1] - constants.STAR_POS[1]
        dist = math.hypot(dx, dy)
        if dist <= constants.INNER_RADIUS:
            target = 1.0
        else:
            target = max(constants.MIN_ZOOM, constants.INNER_RADIUS / dist)
    else:
        target = 1.0
    alpha = 1.0 - math.exp(-dt * constants.ZOOM_SMOOTHING)
    gs.view_zoom += (target - gs.view_zoom) * alpha


def _fail(gs: GameState, message: str) -> None:
    gs.status = message
    gs.state = State.FAILED


def _captured_stranded(
    gs: GameState, capture_radius: float, dock_speed_limit: float
) -> bool:
    """Returns True iff a state transition occurred (DOCKED on a clean dock,
    FAILED if impact speed exceeded the dock_speed_limit hard fail)."""
    dx = gs.tug_pos[0] - gs.stranded_pos[0]
    dy = gs.tug_pos[1] - gs.stranded_pos[1]
    if dx * dx + dy * dy > capture_radius * capture_radius:
        return False
    stranded_vel = LEVELS[gs.level].stranded_orbit.velocity(gs.stranded_anomaly)
    impact = _relative_speed(gs.tug_vel, stranded_vel)
    gs.rv_stranded_at_lock = impact
    if impact > dock_speed_limit:
        _fail(gs, "Hard dock — stranded vessel lost.")
        return True
    gs.state = State.DOCKED
    gs.status = "Docked!"
    gs.phase_anim_elapsed = 0.0
    gs.dock_tug_start_facing = gs.tug_angle
    # Snapshot the tug's offset from the stranded ship's cargo-mount point
    # at this instant; the DOCKED animation rides the live stranded orbit
    # and decays this offset to zero. Avoids the bouncing that came from
    # lerping toward an imagined circular orbit around STAR_POS while the
    # stranded ship moved along its real (possibly elliptical or
    # lemniscate) path.
    target = tug_visual_center(gs.stranded_pos, constants.STRANDED_FACING, cargo=True)
    gs.dock_tug_offset_at_capture = (
        gs.tug_pos[0] - target[0],
        gs.tug_pos[1] - target[1],
    )
    gs.has_cargo = True
    return True


def _captured_rescue(
    gs: GameState, capture_radius: float, dock_speed_limit: float
) -> bool:
    """Returns True iff a state transition occurred (RETURNING on a clean
    dock, FAILED if impact speed exceeded the dock_speed_limit hard fail)."""
    rdx = gs.tug_pos[0] - constants.RESCUE_POS[0]
    rdy = gs.tug_pos[1] - constants.RESCUE_POS[1]
    if rdx * rdx + rdy * rdy > capture_radius * capture_radius:
        return False
    impact = _relative_speed(gs.tug_vel, (0.0, 0.0))
    gs.rv_rescue_at_lock = impact
    gs.rv_rescue_locked = True
    if impact > dock_speed_limit:
        _fail(gs, "Hard dock — tug crashed on return.")
        return True
    gs.state = State.RETURNING
    gs.status = "Returning to rescue ship..."
    gs.phase_anim_elapsed = 0.0
    gs.dock_tug_start_facing = gs.tug_angle
    gs.dock_stranded_start_pos = gs.stranded_pos
    gs.dock_tug_start_pos = tug_visual_center(
        gs.tug_pos, gs.tug_angle, cargo=True
    )
    return True


def _advance_stranded(gs: GameState, dt: float) -> None:
    orbit = LEVELS[gs.level].stranded_orbit
    gs.stranded_anomaly = orbit.advance(gs.stranded_anomaly, dt)
    gs.stranded_pos = orbit.position(gs.stranded_anomaly, constants.STAR_POS)


def _settings_lines(
    level: int,
    difficulty: str,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Settings readout: level + cyclable [d]ifficulty plus its three derived values."""
    preset = constants.DIFFICULTY_PRESETS[difficulty]
    capture = str(preset["capture"])
    rv_setting = str(preset["rv_max"])
    rv_limit = constants.DOCK_SPEED_LIMITS[rv_setting]
    damage_mult = float(preset["damage_mult"])
    level_keys_label = "/".join(str(k) for k in LEVEL_KEYS)
    return [
        [
            (f"[{level_keys_label}]", constants.HUD_ACTION),
            (f" level: {level} ({LEVELS[level].name})", constants.HUD),
        ],
        [("[d]", constants.HUD_ACTION), (f"ifficulty: {difficulty}", constants.HUD)],
        [("  capture radius: ", constants.HUD), (capture, constants.HUD_ACTION)],
        [("  rv max: ", constants.HUD), (f"{rv_limit:.0f} px/s", constants.HUD_ACTION)],
        [("  damage rate: ", constants.HUD), (f"{damage_mult:.1f}x", constants.HUD_ACTION)],
    ]


def _format_mission_time(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:04.1f}s" if minutes else f"{secs:.1f}s"


RESCUE_REPORT_TITLE = "RESCUE COMPLETE"


def _rescue_report_body(
    gs: GameState,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Body of the win-screen modal: mission time (which is the score),
    the two locked dock relative speeds, fuel used, and per-ship damage."""
    fuel_used_pct = (constants.FUEL_CAPACITY - gs.fuel) / constants.FUEL_CAPACITY * 100
    return [
        [("score: ", constants.HUD), (_format_mission_time(gs.mission_elapsed), constants.HUD_OK)],
        [],
        [
            ("rv at ", constants.HUD),
            ("stranded", constants.STRANDED),
            (f" dock: {gs.rv_stranded_at_lock:5.1f} px/s", constants.HUD_ACTION),
        ],
        [
            ("rv at ", constants.HUD),
            ("rescue", constants.RESCUE),
            (f" dock: {gs.rv_rescue_at_lock:5.1f} px/s", constants.HUD_ACTION),
        ],
        [],
        [("fuel used: ", constants.HUD), (f"{int(round(fuel_used_pct))}%", constants.HUD_ACTION)],
        [
            ("damage to ", constants.HUD),
            ("tug", constants.TUG),
            (f": {int(round(gs.tug_damage))}%", constants.HUD_ACTION),
        ],
        [
            ("damage to ", constants.HUD),
            ("stranded", constants.STRANDED),
            (f": {int(round(gs.stranded_damage))}%", constants.HUD_ACTION),
        ],
    ]


def _state_action_prompts(
    gs: GameState,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Contextual [SPACE] prompts to advance state. Empty for states with
    no available player action (OUTBOUND/HOMEBOUND/WON/FAILED/RETURNING)."""
    if gs.state is State.ARRIVING:
        return [[("[SPACE]", constants.HUD_ACTION), (" skip intro", constants.HUD)]]
    if gs.state is State.PARKED:
        return [[("[SPACE]", constants.HUD_ACTION), (" launch tug", constants.HUD)]]
    if gs.state is State.DOCKED and gs.phase_anim_elapsed >= constants.PHASE_ANIM_DURATION:
        return [[("[SPACE]", constants.HUD_ACTION), (" begin return journey", constants.HUD)]]
    return []


def _panel_max_chars(font: pygame.font.Font) -> int:
    """Characters per line that fit inside the side panel at the given
    monospace font. 20px padding on each side, 1px buffer to avoid bleeding
    into the divider."""
    char_w = font.size("M")[0] or 1
    return max(1, (constants.SIDE_PANEL_WIDTH - 40 - 1) // char_w)


# Inside-the-box content padding for boxed sections.
_BOX_PAD_X = 10
_BOX_PAD_TOP = 12
_BOX_PAD_BOT = 8


def _draw_section_frame(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    label: str,
    label_color: tuple[int, int, int],
) -> None:
    """1-px box around `rect` with the label inset into the top edge.

    The label sits centered on the top line (half above, half below) and the
    line breaks around it for a small 'bump out' tab look. Side and bottom
    edges are continuous."""
    border = constants.PANEL_DIVIDER
    label_surf = font.render(label, True, label_color)
    lw, lh = label_surf.get_width(), label_surf.get_height()
    label_x = rect.x + 14
    label_y = rect.y - lh // 2
    label_pad = 6

    # Left, right, bottom edges
    pygame.draw.line(screen, border, (rect.x, rect.y), (rect.x, rect.bottom - 1))
    pygame.draw.line(screen, border, (rect.right - 1, rect.y), (rect.right - 1, rect.bottom - 1))
    pygame.draw.line(screen, border, (rect.x, rect.bottom - 1), (rect.right - 1, rect.bottom - 1))
    # Top edge — broken around the label
    pygame.draw.line(screen, border, (rect.x, rect.y), (label_x - label_pad, rect.y))
    pygame.draw.line(screen, border, (label_x + lw + label_pad, rect.y), (rect.right - 1, rect.y))
    screen.blit(label_surf, (label_x, label_y))


def _box_inner(rect: pygame.Rect) -> tuple[int, int, int]:
    """Top-left x, top-left y, and inner width for content inside a section frame."""
    return (
        rect.x + _BOX_PAD_X,
        rect.y + _BOX_PAD_TOP,
        rect.width - 2 * _BOX_PAD_X,
    )


def _draw_kv(
    screen: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    key_segs: list[tuple[str, tuple[int, int, int]]],
    value_segs: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Left-aligned key, right-aligned value at width `w`."""
    key_s = render.render_key_line(font, key_segs)
    val_s = render.render_key_line(font, value_segs)
    screen.blit(key_s, (x, y))
    screen.blit(val_s, (x + w - val_s.get_width(), y))


def _draw_meter(
    screen: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    label: str,
    label_color: tuple[int, int, int],
    fill_ratio: float,
    fill_color: tuple[int, int, int],
    value_text: str,
) -> None:
    """Sensor-style meter: `label` left, bar in the middle, `value_text` right."""
    label_s = font.render(label, True, label_color)
    value_s = font.render(value_text, True, constants.HUD_ACTION)
    bar_h = 8
    bar_pad = 8
    bar_x = x + label_s.get_width() + bar_pad
    bar_right = x + w - value_s.get_width() - bar_pad
    bar_w = max(0, bar_right - bar_x)
    bar_y = y + (label_s.get_height() - bar_h) // 2
    screen.blit(label_s, (x, y))
    screen.blit(value_s, (x + w - value_s.get_width(), y))
    if bar_w > 0:
        pygame.draw.rect(screen, constants.PANEL_DIVIDER, (bar_x, bar_y, bar_w, bar_h), 1)
        inner_w = max(0, int((bar_w - 2) * max(0.0, min(1.0, fill_ratio))))
        if inner_w > 0:
            pygame.draw.rect(screen, fill_color, (bar_x + 1, bar_y + 1, inner_w, bar_h - 2))


def _draw_telemetry_section(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    gs: GameState,
) -> None:
    """Sensor readouts: time, two relative velocities, fuel + two damage meters, star state."""
    _draw_section_frame(screen, rect, font, "TELEMETRY", constants.PANEL_HEADER)
    x, y, w = _box_inner(rect)
    line_h = font.get_height() + 2

    _draw_kv(
        screen, font, x, y, w,
        [("time", constants.HUD)],
        [(_format_mission_time(gs.mission_elapsed), constants.HUD_ACTION)],
    )
    y += line_h

    if gs.has_cargo:
        rv_s, rv_s_color = gs.rv_stranded_at_lock, constants.HUD_ACTION
    else:
        sv = LEVELS[gs.level].stranded_orbit.velocity(gs.stranded_anomaly)
        rv_s = _relative_speed(gs.tug_vel, sv)
        rv_s_color = constants.HUD
    _draw_kv(
        screen, font, x, y, w,
        [("rv ", constants.HUD), ("stranded", constants.STRANDED)],
        [(f"{rv_s:6.1f} px/s", rv_s_color)],
    )
    y += line_h

    if gs.rv_rescue_locked:
        rv_r, rv_r_color = gs.rv_rescue_at_lock, constants.HUD_ACTION
    else:
        rv_r = _relative_speed(gs.tug_vel, (0.0, 0.0))
        rv_r_color = constants.HUD
    _draw_kv(
        screen, font, x, y, w,
        [("rv ", constants.HUD), ("rescue", constants.RESCUE)],
        [(f"{rv_r:6.1f} px/s", rv_r_color)],
    )
    y += line_h

    y += 8  # spacer between numerical readouts and meters

    # Labels padded to 15 chars so all three bars start at the same x. The
    # label color identifies which ship the meter belongs to; the bar fill
    # color indicates ok/bad state.
    fuel_pct = max(0.0, min(1.0, gs.fuel / constants.FUEL_CAPACITY))
    _draw_meter(
        screen, font, x, y, w, "tug fuel       ", constants.TUG,
        fuel_pct, constants.HUD_OK, f"{int(round(fuel_pct * 100)):3d}%",
    )
    y += line_h
    tug_pct = max(0.0, min(1.0, gs.tug_damage / constants.DAMAGE_CAPACITY))
    _draw_meter(
        screen, font, x, y, w, "tug damage     ", constants.TUG,
        tug_pct, constants.HUD_BAD, f"{int(round(tug_pct * 100)):3d}%",
    )
    y += line_h
    str_pct = max(0.0, min(1.0, gs.stranded_damage / constants.DAMAGE_CAPACITY))
    _draw_meter(
        screen, font, x, y, w, "stranded damage", constants.STRANDED,
        str_pct, constants.HUD_BAD, f"{int(round(str_pct * 100)):3d}%",
    )
    y += line_h

    # Star state: dash on flare-disabled levels; otherwise countdown,
    # then FLARES once it triggers.
    in_mission = gs.state in (State.OUTBOUND, State.DOCKED, State.HOMEBOUND, State.RETURNING)
    if not LEVELS[gs.level].flares:
        val_text, val_color = "—", constants.HUD
    elif in_mission and gs.mission_elapsed < constants.INSTABILITY_DELAY:
        countdown = constants.INSTABILITY_DELAY - gs.mission_elapsed
        val_text, val_color = f"T-{countdown:4.1f}s", constants.HUD_ACTION
    elif _flares_active(gs):
        val_text, val_color = "FLARES", constants.HUD_BAD
    else:
        val_text, val_color = "stable", constants.HUD
    _draw_kv(
        screen, font, x, y, w,
        [("star    ", constants.HUD)],
        [(val_text, val_color)],
    )


def _mission_section_content(
    gs: GameState, briefing_visible: bool
) -> tuple[
    str,
    tuple[int, int, int],
    list[list[tuple[str, tuple[int, int, int]]]],
    list[tuple[str, tuple[int, int, int]]],
]:
    """Pick header label/color, body lines, and prompt for the mission section."""
    if briefing_visible:
        return (
            "MISSION BRIEFING",
            constants.HUD_ACTION,
            BRIEFING_LINES,
            [("[SPACE]", constants.HUD_ACTION), (" to continue", constants.HUD)],
        )
    if gs.state is State.WON:
        return (
            "RESCUE COMPLETE",
            constants.HUD_OK,
            _rescue_report_body(gs),
            [("[R]", constants.HUD_ACTION), (" to play again", constants.HUD)],
        )
    if gs.state is State.FAILED:
        return (
            "MISSION FAILED",
            constants.HUD_BAD,
            [[(gs.status, constants.HUD)]],
            [("[R]", constants.HUD_ACTION), (" to try again", constants.HUD)],
        )
    body: list[list[tuple[str, tuple[int, int, int]]]] = [[(gs.status, constants.HUD)]]
    prompts = _state_action_prompts(gs)
    prompt = prompts[0] if prompts else []
    return ("STATUS", constants.PANEL_HEADER, body, prompt)


def _draw_mission_section(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    gs: GameState,
    briefing_visible: bool,
) -> None:
    """Dedicated text area for briefing / rescue report / failure / status.
    The frame label is state-colored so the section reads at a glance."""
    label, color, body_lines, prompt = _mission_section_content(gs, briefing_visible)
    _draw_section_frame(screen, rect, font, label, color)
    x, y, w = _box_inner(rect)
    body_line_h = font.get_height() + 2
    char_w = font.size("M")[0] or 1
    max_chars = max(1, w // char_w)
    for line_segs in body_lines:
        for wrapped in render.wrap_segments(line_segs, max_chars):
            if wrapped:
                s = render.render_key_line(font, wrapped)
                screen.blit(s, (x, y))
            y += body_line_h
    if prompt:
        y += 4
        for wrapped in render.wrap_segments(prompt, max_chars):
            if wrapped:
                s = render.render_key_line(font, wrapped)
                screen.blit(s, (x, y))
            y += body_line_h


def _draw_settings_section(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    level: int,
    difficulty: str,
) -> None:
    _draw_section_frame(screen, rect, font, "SETTINGS", constants.PANEL_HEADER)
    x, y, _w = _box_inner(rect)
    line_h = font.get_height() + 2
    for segs in _settings_lines(level, difficulty):
        s = render.render_key_line(font, segs)
        screen.blit(s, (x, y))
        y += line_h


def _draw_records_section(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    records: dict[tuple[int, str], float],
    active_level: int,
    active_difficulty: str,
) -> None:
    """Best mission time per difficulty for the active level. The active
    difficulty cell is colored HUD_ACTION; unrecorded cells show dashes."""
    _draw_section_frame(screen, rect, font, "RECORDS", constants.PANEL_HEADER)
    x, y, w = _box_inner(rect)

    # Three evenly-spaced cells, one per difficulty. Each cell renders as
    # "d<n> <time>" with the label dimmer than the value so the eye lands
    # on the times.
    n_diff = len(constants.DIFFICULTY_CYCLE)
    cell_w = w // n_diff
    for i, d in enumerate(constants.DIFFICULTY_CYCLE):
        best = records.get((active_level, d))
        is_active = d == active_difficulty
        if best is None:
            time_text, time_color = "--", constants.HUD
        else:
            time_text = _format_mission_time(best)
            time_color = constants.HUD_ACTION if is_active else constants.HUD_OK
        label_color = constants.HUD_ACTION if is_active else constants.PANEL_HEADER
        segs: list[tuple[str, tuple[int, int, int]]] = [
            (f"d{d} ", label_color),
            (time_text, time_color),
        ]
        line = render.render_key_line(font, segs)
        cell_x = x + i * cell_w + (cell_w - line.get_width()) // 2
        screen.blit(line, (cell_x, y))


def _draw_controls_section(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
) -> None:
    _draw_section_frame(screen, rect, font, "CONTROLS", constants.PANEL_HEADER)
    x, y, w = _box_inner(rect)
    line_h = font.get_height() + 2
    col2_x = x + w // 2 + 4
    rows = [
        (
            [("[LEFT/RIGHT]", constants.HUD_ACTION), (" rotate", constants.HUD)],
            [("[R]", constants.HUD_ACTION), ("   reset", constants.HUD)],
        ),
        (
            [("[UP]", constants.HUD_ACTION), (" thrust", constants.HUD)],
            [("[ESC]", constants.HUD_ACTION), (" quit", constants.HUD)],
        ),
    ]
    for left_segs, right_segs in rows:
        screen.blit(render.render_key_line(font, left_segs), (x, y))
        screen.blit(render.render_key_line(font, right_segs), (col2_x, y))
        y += line_h


def _draw_side_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    gs: GameState,
    difficulty: str,
    briefing_visible: bool,
    records: dict[tuple[int, str], float],
) -> None:
    panel_w = constants.SIDE_PANEL_WIDTH
    panel_h = constants.WINDOW_SIZE[1]
    pygame.draw.rect(screen, constants.PANEL_BG, (0, 0, panel_w, panel_h))
    pygame.draw.line(
        screen, constants.PANEL_DIVIDER,
        (panel_w - 1, 0), (panel_w - 1, panel_h - 1), 1,
    )

    pad = 18
    x = pad
    inner_w = panel_w - 2 * pad
    line_h = font.get_height() + 2

    # Each boxed section: 1 px frame + _BOX_PAD_TOP + content + _BOX_PAD_BOT + 1 px frame.
    box_overhead = _BOX_PAD_TOP + _BOX_PAD_BOT + 2

    # Telemetry: 7 content lines (time, two rv, three meters, star) + 8 px
    # spacer between the numerical readouts and the meter bars. Settings
    # has 5 lines (level + difficulty + three derived readouts). Records
    # has 1 line — three difficulty cells for the active level.
    telemetry_h = box_overhead + line_h * 7 + 8
    settings_h = box_overhead + line_h * 5
    controls_h = box_overhead + line_h * 2
    records_h = box_overhead + line_h * 1
    section_gap = 16

    # Mission box fills the slack at the top so the briefing/report has plenty
    # of room and the lower sections are fixed-position.
    fixed_total = telemetry_h + settings_h + controls_h + records_h + 4 * section_gap
    mission_h = panel_h - 2 * pad - fixed_total

    y = pad
    mission_rect = pygame.Rect(x, y, inner_w, mission_h)
    _draw_mission_section(screen, mission_rect, font, gs, briefing_visible)
    y += mission_h + section_gap

    telemetry_rect = pygame.Rect(x, y, inner_w, telemetry_h)
    _draw_telemetry_section(screen, telemetry_rect, font, gs)
    y += telemetry_h + section_gap

    settings_rect = pygame.Rect(x, y, inner_w, settings_h)
    _draw_settings_section(screen, settings_rect, font, gs.level, difficulty)
    y += settings_h + section_gap

    controls_rect = pygame.Rect(x, y, inner_w, controls_h)
    _draw_controls_section(screen, controls_rect, font)
    y += controls_h + section_gap

    records_rect = pygame.Rect(x, y, inner_w, records_h)
    _draw_records_section(screen, records_rect, font, records, gs.level, difficulty)


def draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    gs: GameState,
    thrusting: bool,
    capture_radius: float,
    difficulty: str,
    briefing_visible: bool,
    records: dict[tuple[int, str], float],
) -> None:
    cam = render.Camera(gs.view_zoom)
    # Clip world rendering to the gameplay viewport so anything that runs
    # off-screen (orbit guides, lost-radius ring, ships at low zoom) stays
    # out of the side panel.
    screen.set_clip(pygame.Rect(*constants.VIEWPORT_RECT))
    level_def = LEVELS[gs.level]
    render.draw_background(screen, cam, level_def.stranded_orbit, level_def.stars)
    growth = _flare_growth(gs)
    if growth > 0.0:
        t_wall = pygame.time.get_ticks() / 1000.0
        render.draw_radiation_zone(screen, cam, t_wall, growth)
    render.draw_trail(screen, gs.tug_trail, cam)
    rescue_pos = rescue_ship_position(gs.arrival_progress)
    render.draw_rescue_ship(screen, rescue_pos, cam)

    # Stranded ship: position and facing depend on state. Default = its own
    # spot with the fixed disabled orientation. RETURNING lerps both into the
    # rescue position and rescue facing so the three ships overlap on landing.
    stranded_draw_pos: tuple[float, float] = gs.stranded_pos
    stranded_facing = constants.STRANDED_FACING

    # Tug position/facing/thrust depend on state.
    show_halo = True
    if gs.state in (State.ARRIVING, State.PARKED):
        # Nested inside the rescue ship, facing outward from the star.
        dx = constants.STAR_POS[0] - rescue_pos[0]
        dy = constants.STAR_POS[1] - rescue_pos[1]
        tug_facing = math.atan2(dy, dx) + math.pi
        tug_visual_pos = rescue_pos
        tug_thrust = False
    elif gs.state is State.DOCKED:
        # Game takes control. The tug rides along the live stranded orbit
        # (which `_advance_stranded` is updating each tick), and the
        # offset captured at dock smoothly decays to zero — so by the end
        # of the animation the tug sits exactly on the stranded ship's
        # cargo mount, regardless of orbit shape or gravity field.
        elapsed = gs.phase_anim_elapsed
        target_pos = tug_visual_center(gs.stranded_pos, constants.STRANDED_FACING, cargo=True)
        t_raw = min(1.0, elapsed / constants.PHASE_ANIM_DURATION)
        t = t_raw * t_raw * (3 - 2 * t_raw)
        ox, oy = gs.dock_tug_offset_at_capture
        tug_visual_pos = (
            target_pos[0] + ox * (1 - t),
            target_pos[1] + oy * (1 - t),
        )
        tug_facing = _lerp_angle(gs.dock_tug_start_facing, constants.STRANDED_FACING, t)
        tug_thrust = False
        if t_raw < 1.0:
            show_halo = int(elapsed / 0.2) % 2 == 0
    elif gs.state is State.RETURNING:
        # Game takes control. Both ships glide in straight lines from their
        # collision-moment positions to the rescue ship and rotate to align
        # with its outward facing (0). No orbital motion at this radius.
        elapsed = gs.phase_anim_elapsed
        target = (float(constants.RESCUE_POS[0]), float(constants.RESCUE_POS[1]))
        t_raw = min(1.0, elapsed / constants.PHASE_ANIM_DURATION)
        t = t_raw * t_raw * (3 - 2 * t_raw)
        stranded_draw_pos = (
            gs.dock_stranded_start_pos[0] * (1 - t) + target[0] * t,
            gs.dock_stranded_start_pos[1] * (1 - t) + target[1] * t,
        )
        stranded_facing = _lerp_angle(constants.STRANDED_FACING, 0.0, t)
        tug_visual_pos = (
            gs.dock_tug_start_pos[0] * (1 - t) + target[0] * t,
            gs.dock_tug_start_pos[1] * (1 - t) + target[1] * t,
        )
        tug_facing = _lerp_angle(gs.dock_tug_start_facing, 0.0, t)
        tug_thrust = False
        if t_raw < 1.0:
            show_halo = int(elapsed / 0.2) % 2 == 0
    else:
        # OUTBOUND, HOMEBOUND, WON, FAILED — driven by physics or pilot.
        tug_facing = gs.tug_angle
        # WON ends with everything overlapping the rescue ship — no offset
        # and stranded aligned to the rescue ship's outward facing (0).
        cargo = gs.has_cargo and gs.state is not State.WON
        tug_visual_pos = tug_visual_center(gs.tug_pos, tug_facing, cargo=cargo)
        # Flame only renders when thrust is actually being applied — gated
        # on fuel so an empty tank shows no flame even if UP is held.
        tug_thrust = thrusting and gs.fuel > 0.0
        if gs.state is State.WON:
            stranded_facing = 0.0

    render.draw_stranded(screen, stranded_draw_pos, cam, facing=stranded_facing)
    # Halo represents the capture zone — must be centered on the capture-check
    # origin. During OUTBOUND/HOMEBOUND that's gs.tug_pos (= cargo center when
    # towing); for animations (DOCKED/RETURNING) and parked states the halo
    # is decorative so tug_visual_pos is fine.
    halo_pos = gs.tug_pos if gs.state in (State.OUTBOUND, State.HOMEBOUND) else tug_visual_pos
    if show_halo:
        render.draw_capture_halo(screen, halo_pos, capture_radius, cam)
    render.draw_tug(screen, tug_visual_pos, tug_facing, thrusting=tug_thrust, cargo=False, cam=cam)

    # Dock points: small dots at the centers used by the capture check so the
    # geometry of "halo encloses point" is visible. Stranded dock point hides
    # once the vessel is in tow (no longer a docking target).
    if not gs.has_cargo:
        render.draw_dock_point(screen, gs.stranded_pos, cam)
    if gs.state is not State.WON:
        render.draw_dock_point(screen, rescue_pos, cam)

    screen.set_clip(None)

    _draw_side_panel(screen, font, title_font, gs, difficulty, briefing_visible, records)


# Number-key bindings for fast level switching. K_1 → level 1, K_2 → level 2, …
_LEVEL_KEY_MAP: dict[int, int] = {
    getattr(pygame, f"K_{n}"): n for n in LEVEL_KEYS
}


async def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(constants.WINDOW_SIZE)
    pygame.display.set_caption("Orbital Rescue")
    clock = pygame.time.Clock()
    font_path = str(Path(__file__).parent / "MonaspaceNeon-Regular.otf")
    font = pygame.font.Font(font_path, 14)
    title_font = pygame.font.Font(font_path, 22)

    level = 1
    difficulty = "1"
    briefing_dismissed = False
    gs = GameState.new(level=level, with_arrival=True)
    # Per-(level, difficulty) best mission times. In-memory only — records
    # reset on process exit / page refresh.
    records: dict[tuple[int, str], float] = {}
    # Tracks whether the latest WON state has already been written to records.
    # Set False on every reset / level switch.
    win_recorded = False
    accumulator = 0.0

    running = True
    while running:
        real_dt = clock.tick(constants.FPS) / 1000.0
        accumulator += min(real_dt, constants.MAX_FRAME_DT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Shift+R: hard reset to the start (arrival + briefing
                    # again). Plain R: quick retry from PARKED. Shift+R is
                    # intentionally not surfaced in the controls block —
                    # it's a debugging convenience.
                    hard = bool(event.mod & pygame.KMOD_SHIFT)
                    gs = GameState.new(level=level, with_arrival=hard)
                    briefing_dismissed = not hard
                    win_recorded = False
                    accumulator = 0.0
                elif event.key == pygame.K_d:
                    i = constants.DIFFICULTY_CYCLE.index(difficulty)
                    difficulty = constants.DIFFICULTY_CYCLE[(i + 1) % len(constants.DIFFICULTY_CYCLE)]
                elif event.key in _LEVEL_KEY_MAP:
                    level = _LEVEL_KEY_MAP[event.key]
                    gs = GameState.new(level=level, with_arrival=False)
                    briefing_dismissed = True
                    win_recorded = False
                    accumulator = 0.0
                elif event.key == pygame.K_SPACE:
                    if gs.state is State.ARRIVING:
                        gs.finish_arrival()
                    elif gs.state is State.PARKED:
                        if not briefing_dismissed:
                            briefing_dismissed = True
                        else:
                            gs.launch_outbound()
                    elif gs.state is State.DOCKED:
                        if gs.phase_anim_elapsed >= constants.PHASE_ANIM_DURATION:
                            gs.engage_homebound()

        keys = pygame.key.get_pressed()
        piloting = gs.state in (State.OUTBOUND, State.HOMEBOUND)
        inp = InputFrame(
            left=piloting and keys[pygame.K_LEFT],
            right=piloting and keys[pygame.K_RIGHT],
            thrust=piloting and keys[pygame.K_UP],
        )
        preset = constants.DIFFICULTY_PRESETS[difficulty]
        capture_radius = constants.CAPTURE_RADII[str(preset["capture"])]
        dock_speed_limit = constants.DOCK_SPEED_LIMITS[str(preset["rv_max"])]
        damage_mult = float(preset["damage_mult"])

        while accumulator >= constants.DT_SIM:
            simulate(gs, constants.DT_SIM, inp, capture_radius, damage_mult, dock_speed_limit)
            accumulator -= constants.DT_SIM

        if not win_recorded and gs.state is State.WON:
            key = (level, difficulty)
            prev = records.get(key)
            if prev is None or gs.mission_elapsed < prev:
                records[key] = gs.mission_elapsed
            win_recorded = True

        briefing_visible = gs.state is State.PARKED and not briefing_dismissed
        draw(screen, font, title_font, gs, inp.thrust, capture_radius, difficulty, briefing_visible, records)
        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
