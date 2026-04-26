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
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import pygame

import constants
from geometry import tug_visual_center
from physics import (
    circular_orbit_angular_speed,
    circular_orbit_position,
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


def off_screen(pos: tuple[float, float], margin: float = 400.0) -> bool:
    return (
        pos[0] < -margin
        or pos[0] > constants.WINDOW_SIZE[0] + margin
        or pos[1] < -margin
        or pos[1] > constants.WINDOW_SIZE[1] + margin
    )


def tug_launch_orbit() -> tuple[tuple[float, float], tuple[float, float], float]:
    """Position, velocity, and facing of the tug on the rescue ship's circular orbit.

    Facing matches the orientation the tug had while nested inside the rescue
    ship — pointing outward from the star — so launching produces no snap.
    """
    pos = (float(constants.RESCUE_POS[0]), float(constants.RESCUE_POS[1]))
    vel = circular_orbit_velocity(constants.RESCUE_ORBIT_RADIUS, 0.0)
    facing = math.atan2(constants.RESCUE_POS[1] - constants.STAR_POS[1], constants.RESCUE_POS[0] - constants.STAR_POS[0])
    return pos, vel, facing


def random_stranded_angle() -> float:
    return random.uniform(0.0, 2 * math.pi)


BRIEFING_TITLE = "MISSION BRIEFING"
BRIEFING_LINES: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("A ", constants.HUD), ("stranded", constants.STRANDED), (" vessel drifts in low orbit around", constants.HUD)],
    [("a hostile star, taking radiation damage every", constants.HUD)],
    [("second it remains.", constants.HUD)],
    [],
    [("Your ", constants.HUD), ("rescue", constants.RESCUE), (" ship holds a safer orbit above.", constants.HUD)],
    [("Launch the hardened ", constants.HUD), ("tug", constants.TUG), (" down to the ", constants.HUD), ("stranded", constants.STRANDED)],
    [("ship, dock, and haul it back to safety.", constants.HUD)],
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


def _impact_speed(
    tug_vel: tuple[float, float], target_vel: tuple[float, float]
) -> float:
    """Magnitude of the tug-vs-target relative velocity — i.e., the impact
    speed if the two collided right now. Direction-agnostic; low = smooth,
    high = rough."""
    return math.hypot(tug_vel[0] - target_vel[0], tug_vel[1] - target_vel[1])


@dataclass
class GameState:
    state: State
    tug_pos: tuple[float, float]
    tug_vel: tuple[float, float]
    tug_angle: float
    tug_trail: deque[tuple[float, float]]
    pilot_age: float
    stranded_angle: float
    stranded_pos: tuple[float, float]
    arrival_progress: float
    # Shared timer for the two game-driven animations: the DOCKED rotate-to-
    # prograde and the RETURNING glide-into-rescue. They never overlap.
    phase_anim_elapsed: float
    dock_tug_start_facing: float
    dock_tug_capture_radius: float
    dock_tug_capture_angle: float
    dock_tug_start_pos: tuple[float, float]
    dock_stranded_start_pos: tuple[float, float]
    dv_stranded_at_lock: float
    dv_rescue_at_lock: float
    dv_rescue_locked: bool
    has_cargo: bool
    status: str

    @classmethod
    def new(cls, *, with_arrival: bool = False) -> "GameState":
        """Fresh game: tug parked on the rescue ship, stranded ship at a random phase.

        `with_arrival=True` runs the intro animation; reset (R) uses the
        default so the player jumps straight to the settled state.
        """
        pos, _vel, facing = tug_launch_orbit()
        stranded_angle = random_stranded_angle()
        return cls(
            state=State.ARRIVING if with_arrival else State.PARKED,
            tug_pos=pos,
            tug_vel=(0.0, 0.0),
            tug_angle=facing,
            tug_trail=deque(maxlen=constants.MAX_TRAIL_POINTS),
            pilot_age=0.0,
            stranded_angle=stranded_angle,
            stranded_pos=circular_orbit_position(
                constants.STAR_POS, constants.STRANDED_ORBIT_RADIUS, stranded_angle
            ),
            arrival_progress=0.0 if with_arrival else 1.0,
            phase_anim_elapsed=0.0,
            dock_tug_start_facing=0.0,
            dock_tug_capture_radius=0.0,
            dock_tug_capture_angle=0.0,
            dock_tug_start_pos=(0.0, 0.0),
            dock_stranded_start_pos=(0.0, 0.0),
            dv_stranded_at_lock=0.0,
            dv_rescue_at_lock=0.0,
            dv_rescue_locked=False,
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
        pos, vel, facing = tug_launch_orbit()
        self.tug_pos = pos
        self.tug_vel = vel
        self.tug_angle = facing
        self.tug_trail.clear()
        self.tug_trail.append(pos)
        self.pilot_age = 0.0
        self.state = State.OUTBOUND
        self.status = "Pilot the tug to the stranded ship."

    def engage_homebound(self) -> None:
        vel = circular_orbit_velocity(constants.STRANDED_ORBIT_RADIUS, self.stranded_angle)
        self.tug_pos = self.stranded_pos
        self.tug_vel = vel
        # Preserve the post-alignment orientation so HOMEBOUND starts with no snap.
        self.tug_angle = constants.STRANDED_FACING
        self.tug_trail.clear()
        self.tug_trail.append(self.tug_pos)
        self.pilot_age = 0.0
        self.state = State.HOMEBOUND
        self.status = "Piloting home with cargo."


def simulate(
    gs: GameState,
    dt: float,
    inp: InputFrame,
    capture_radius: float,
    stranded_omega: float,
) -> None:
    """Advance the game world by one fixed simulation step (`dt` seconds)."""
    if gs.state is State.ARRIVING:
        _tick_arriving(gs, dt, stranded_omega)
    elif gs.state is State.PARKED:
        _advance_stranded(gs, dt, stranded_omega)
    elif gs.state is State.OUTBOUND:
        _tick_outbound(gs, dt, inp, capture_radius, stranded_omega)
    elif gs.state is State.DOCKED:
        _tick_docked(gs, dt, stranded_omega)
    elif gs.state is State.HOMEBOUND:
        _tick_homebound(gs, dt, inp, capture_radius)
    elif gs.state is State.RETURNING:
        _tick_returning(gs, dt)
    # WON, FAILED — no-op


def _tick_arriving(gs: GameState, dt: float, stranded_omega: float) -> None:
    _advance_stranded(gs, dt, stranded_omega)
    gs.arrival_progress += dt / constants.ARRIVAL_DURATION
    if gs.arrival_progress >= 1.0:
        gs.finish_arrival()


def _tick_outbound(
    gs: GameState,
    dt: float,
    inp: InputFrame,
    capture_radius: float,
    stranded_omega: float,
) -> None:
    _advance_stranded(gs, dt, stranded_omega)
    _apply_pilot_input(gs, dt, inp)
    if _hit_star(gs):
        _fail(gs, "Crashed the tug into the star. (R to reset)")
        return
    if _captured_stranded(gs, capture_radius):
        return
    if _piloting_lost(gs):
        _fail(gs, "Tug lost in space. (R to reset)")


def _tick_docked(gs: GameState, dt: float, stranded_omega: float) -> None:
    _advance_stranded(gs, dt, stranded_omega)
    gs.phase_anim_elapsed += dt


def _tick_homebound(
    gs: GameState, dt: float, inp: InputFrame, capture_radius: float
) -> None:
    _apply_pilot_input(gs, dt, inp)
    gs.stranded_pos = gs.tug_pos
    if _hit_star(gs):
        _fail(gs, "Crashed on the way home. (R to reset)")
        return
    if _captured_rescue(gs, capture_radius):
        return
    if _piloting_lost(gs):
        _fail(gs, "Lost on the way home. (R to reset)")


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
    thrust = (
        (math.cos(gs.tug_angle) * constants.TUG_THRUST, math.sin(gs.tug_angle) * constants.TUG_THRUST)
        if inp.thrust
        else (0.0, 0.0)
    )
    gs.tug_pos, gs.tug_vel = step(gs.tug_pos, gs.tug_vel, constants.STAR_POS, dt, thrust)
    gs.tug_trail.append(gs.tug_pos)
    gs.pilot_age += dt


def _hit_star(gs: GameState) -> bool:
    pdx = gs.tug_pos[0] - constants.STAR_POS[0]
    pdy = gs.tug_pos[1] - constants.STAR_POS[1]
    return pdx * pdx + pdy * pdy <= constants.STAR_RADIUS * constants.STAR_RADIUS


def _piloting_lost(gs: GameState) -> bool:
    return off_screen(gs.tug_pos) or gs.pilot_age >= constants.MAX_PILOT_SECONDS


def _fail(gs: GameState, message: str) -> None:
    gs.status = message
    gs.state = State.FAILED


def _captured_stranded(gs: GameState, capture_radius: float) -> bool:
    dx = gs.tug_pos[0] - gs.stranded_pos[0]
    dy = gs.tug_pos[1] - gs.stranded_pos[1]
    if dx * dx + dy * dy > capture_radius * capture_radius:
        return False
    stranded_vel = circular_orbit_velocity(constants.STRANDED_ORBIT_RADIUS, gs.stranded_angle)
    gs.dv_stranded_at_lock = _impact_speed(gs.tug_vel, stranded_vel)
    gs.state = State.DOCKED
    gs.status = "Docked!"
    gs.phase_anim_elapsed = 0.0
    gs.dock_tug_start_facing = gs.tug_angle
    tug_dx = gs.tug_pos[0] - constants.STAR_POS[0]
    tug_dy = gs.tug_pos[1] - constants.STAR_POS[1]
    gs.dock_tug_capture_radius = math.hypot(tug_dx, tug_dy)
    gs.dock_tug_capture_angle = math.atan2(tug_dy, tug_dx)
    gs.has_cargo = True
    return True


def _captured_rescue(gs: GameState, capture_radius: float) -> bool:
    rdx = gs.tug_pos[0] - constants.RESCUE_POS[0]
    rdy = gs.tug_pos[1] - constants.RESCUE_POS[1]
    if rdx * rdx + rdy * rdy > capture_radius * capture_radius:
        return False
    gs.dv_rescue_at_lock = _impact_speed(gs.tug_vel, (0.0, 0.0))
    gs.dv_rescue_locked = True
    gs.state = State.RETURNING
    gs.status = "Returning to rescue ship..."
    gs.phase_anim_elapsed = 0.0
    gs.dock_tug_start_facing = gs.tug_angle
    gs.dock_stranded_start_pos = gs.stranded_pos
    gs.dock_tug_start_pos = tug_visual_center(
        gs.tug_pos, gs.tug_angle, cargo=True
    )
    return True


def _advance_stranded(gs: GameState, dt: float, omega: float) -> None:
    gs.stranded_angle = (gs.stranded_angle + omega * dt) % (2 * math.pi)
    gs.stranded_pos = circular_orbit_position(
        constants.STAR_POS, constants.STRANDED_ORBIT_RADIUS, gs.stranded_angle
    )


def _settings_lines(
    capture_setting: str, stellar_damage: bool
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    return [
        [("[c]", constants.HUD_ACTION), (f"apture radius: {capture_setting}", constants.HUD)],
        [("[d]", constants.HUD_ACTION), (f"amage: {'on' if stellar_damage else 'off'}", constants.HUD)],
    ]


def _center_content(
    gs: GameState, briefing_visible: bool
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    if briefing_visible:
        return []
    if gs.state is State.ARRIVING:
        return [[("[SPACE]", constants.HUD_ACTION), (" skip intro", constants.HUD)]]
    if gs.state is State.PARKED:
        return [[("[SPACE]", constants.HUD_ACTION), (" launch tug", constants.HUD)]]
    if gs.state is State.DOCKED:
        lines = [[("docked with ", constants.HUD), ("stranded", constants.STRANDED), (" vessel", constants.HUD)]]
        if gs.phase_anim_elapsed >= constants.PHASE_ANIM_DURATION:
            lines.append(
                [("[SPACE]", constants.HUD_ACTION), (" to begin return journey", constants.HUD)]
            )
        return lines
    color = (
        constants.HUD_OK if gs.state is State.WON
        else constants.HUD_BAD if gs.state is State.FAILED
        else constants.HUD
    )
    return [[(gs.status, color)]]


def _draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    gs: GameState,
    capture_setting: str,
    stellar_damage: bool,
    briefing_visible: bool,
) -> None:
    w, h = screen.get_size()
    pad = 12
    line_h = font.get_height() + 4
    y_bot = h - pad - font.get_height()

    # TOP-LEFT: relative velocities to each ship. Live values render in
    # neutral gray; once locked at capture they freeze in action-yellow.
    top_left_lines: list[list[tuple[str, tuple[int, int, int]]]] = []
    if gs.has_cargo:
        dv_s, dv_s_color = gs.dv_stranded_at_lock, constants.HUD_ACTION
    else:
        sv = circular_orbit_velocity(constants.STRANDED_ORBIT_RADIUS, gs.stranded_angle)
        dv_s = _impact_speed(gs.tug_vel, sv)
        dv_s_color = constants.HUD
    top_left_lines.append(
        [("delta-v ", constants.HUD), ("stranded", constants.STRANDED), (f": {dv_s:5.1f} px/s", dv_s_color)]
    )
    if gs.dv_rescue_locked:
        dv_r, dv_r_color = gs.dv_rescue_at_lock, constants.HUD_ACTION
    else:
        dv_r = _impact_speed(gs.tug_vel, (0.0, 0.0))
        dv_r_color = constants.HUD
    top_left_lines.append(
        [("delta-v ", constants.HUD), ("rescue", constants.RESCUE), (f": {dv_r:5.1f} px/s", dv_r_color)]
    )
    for i, segs in enumerate(top_left_lines):
        s = render.render_key_line(font, segs)
        screen.blit(s, (pad, pad + i * line_h))

    # TOP-RIGHT: difficulty settings (key + display combined)
    for i, segs in enumerate(_settings_lines(capture_setting, stellar_damage)):
        s = render.render_key_line(font, segs)
        screen.blit(s, (w - s.get_width() - pad, pad + i * line_h))

    # BOTTOM-LEFT: constant piloting controls
    for i, segs in enumerate(reversed(PILOTING_CONTROLS)):
        s = render.render_key_line(font, segs)
        screen.blit(s, (pad, y_bot - i * line_h))

    # BOTTOM-RIGHT: constant session controls
    for i, segs in enumerate(reversed(SESSION_CONTROLS)):
        s = render.render_key_line(font, segs)
        screen.blit(s, (w - s.get_width() - pad, y_bot - i * line_h))

    # BOTTOM-CENTER: state-advance prompt, or narrative status (hidden while briefing is up)
    lines = _center_content(gs, briefing_visible)
    n = len(lines)
    for i, segs in enumerate(lines):
        s = render.render_key_line(font, segs)
        y = y_bot - (n - 1 - i) * line_h
        screen.blit(s, ((w - s.get_width()) // 2, y))


def draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    gs: GameState,
    thrusting: bool,
    capture_radius: float,
    capture_setting: str,
    stellar_damage: bool,
    briefing_visible: bool,
) -> None:
    render.draw_background(screen, dim_star=briefing_visible)
    render.draw_trail(screen, gs.tug_trail)
    rescue_pos = rescue_ship_position(gs.arrival_progress)
    render.draw_rescue_ship(screen, rescue_pos)

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
        # Game takes control. The tug is treated as if it slipped into a
        # circular orbit at its capture radius, and we lerp from that
        # orbital motion to the stranded vessel's docked offset.
        elapsed = gs.phase_anim_elapsed
        omega = circular_orbit_angular_speed(gs.dock_tug_capture_radius)
        tug_orbit_angle = gs.dock_tug_capture_angle + omega * elapsed
        tug_orbit_pos = (
            constants.STAR_POS[0] + gs.dock_tug_capture_radius * math.cos(tug_orbit_angle),
            constants.STAR_POS[1] + gs.dock_tug_capture_radius * math.sin(tug_orbit_angle),
        )
        target_pos = tug_visual_center(gs.stranded_pos, constants.STRANDED_FACING, cargo=True)
        t_raw = min(1.0, elapsed / constants.PHASE_ANIM_DURATION)
        t = t_raw * t_raw * (3 - 2 * t_raw)
        tug_facing = _lerp_angle(gs.dock_tug_start_facing, constants.STRANDED_FACING, t)
        tug_visual_pos = (
            tug_orbit_pos[0] * (1 - t) + target_pos[0] * t,
            tug_orbit_pos[1] * (1 - t) + target_pos[1] * t,
        )
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
        tug_thrust = thrusting
        if gs.state is State.WON:
            stranded_facing = 0.0

    render.draw_stranded(screen, stranded_draw_pos, stranded_facing)
    if show_halo:
        render.draw_capture_halo(screen, tug_visual_pos, capture_radius)
    render.draw_tug(screen, tug_visual_pos, tug_facing, thrusting=tug_thrust, cargo=False)

    _draw_hud(screen, font, gs, capture_setting, stellar_damage, briefing_visible)

    if briefing_visible:
        render.draw_briefing_modal(
            screen,
            title_font,
            font,
            BRIEFING_TITLE,
            BRIEFING_LINES,
            [("[SPACE]", constants.HUD_ACTION), (" to continue", constants.HUD)],
        )


async def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(constants.WINDOW_SIZE)
    pygame.display.set_caption("Orbital Rescue")
    clock = pygame.time.Clock()
    font_path = str(Path(__file__).parent / "Moby-Monospace.ttf")
    font = pygame.font.Font(font_path, 18)
    title_font = pygame.font.Font(font_path, 28)

    stranded_omega = circular_orbit_angular_speed(constants.STRANDED_ORBIT_RADIUS)
    capture_setting = "medium"
    stellar_damage = False
    briefing_dismissed = False
    gs = GameState.new(with_arrival=True)
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
                    gs = GameState.new()
                    briefing_dismissed = True
                    accumulator = 0.0
                elif event.key == pygame.K_c:
                    i = constants.CAPTURE_CYCLE.index(capture_setting)
                    capture_setting = constants.CAPTURE_CYCLE[(i + 1) % len(constants.CAPTURE_CYCLE)]
                elif event.key == pygame.K_d:
                    stellar_damage = not stellar_damage
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
        capture_radius = constants.CAPTURE_RADII[capture_setting]

        while accumulator >= constants.DT_SIM:
            simulate(gs, constants.DT_SIM, inp, capture_radius, stranded_omega)
            accumulator -= constants.DT_SIM

        briefing_visible = gs.state is State.PARKED and not briefing_dismissed
        draw(screen, font, title_font, gs, inp.thrust, capture_radius, capture_setting, stellar_damage, briefing_visible)
        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
