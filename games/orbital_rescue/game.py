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

from constants import (
    ARRIVAL_DURATION,
    ARRIVAL_START_RADIUS,
    CAPTURE_CYCLE,
    CAPTURE_RADII,
    DOCK_RADIUS,
    DT_SIM,
    FPS,
    HUD,
    HUD_ACTION,
    HUD_BAD,
    HUD_OK,
    MAX_FRAME_DT,
    MAX_PILOT_SECONDS,
    MAX_TRAIL_POINTS,
    RESCUE,
    RESCUE_ORBIT_RADIUS,
    RESCUE_POS,
    STAR_POS,
    STAR_RADIUS,
    STRANDED,
    STRANDED_ORBIT_RADIUS,
    TUG,
    TUG_ROT_SPEED,
    TUG_THRUST,
    WINDOW_SIZE,
)
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
        or pos[0] > WINDOW_SIZE[0] + margin
        or pos[1] < -margin
        or pos[1] > WINDOW_SIZE[1] + margin
    )


def tug_launch_orbit() -> tuple[tuple[float, float], tuple[float, float], float]:
    """Position, velocity, and facing of the tug on the rescue ship's circular orbit."""
    pos = (float(RESCUE_POS[0]), float(RESCUE_POS[1]))
    vel = circular_orbit_velocity(RESCUE_ORBIT_RADIUS, 0.0)
    facing = math.atan2(vel[1], vel[0])
    return pos, vel, facing


def random_stranded_angle() -> float:
    return random.uniform(0.0, 2 * math.pi)


BRIEFING_TITLE = "MISSION BRIEFING"
BRIEFING_LINES: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("A ", HUD), ("stranded", STRANDED), (" vessel drifts in low orbit around", HUD)],
    [("a hostile star, taking radiation damage every", HUD)],
    [("second it remains.", HUD)],
    [],
    [("Your ", HUD), ("rescue", RESCUE), (" ship holds a safer orbit above.", HUD)],
    [("Launch the hardened ", HUD), ("tug", TUG), (" down to the ", HUD), ("stranded", STRANDED)],
    [("ship, dock, and haul it back to safety.", HUD)],
]

PILOTING_CONTROLS: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("[LEFT/RIGHT]", HUD_ACTION), (" rotate", HUD)],
    [("[UP]", HUD_ACTION), (" thrust", HUD)],
]
SESSION_CONTROLS: list[list[tuple[str, tuple[int, int, int]]]] = [
    [("[R]", HUD_ACTION), (" reset", HUD)],
    [("[ESC]", HUD_ACTION), (" quit", HUD)],
]


def rescue_ship_position(arrival_progress: float) -> tuple[float, float]:
    """Rescue ship's current spot: radial descent from ARRIVAL_START_RADIUS to RESCUE_POS with smoothstep easing."""
    if arrival_progress >= 1.0:
        return (float(RESCUE_POS[0]), float(RESCUE_POS[1]))
    t = max(0.0, arrival_progress)
    s = t * t * (3 - 2 * t)
    radius = ARRIVAL_START_RADIUS + (RESCUE_ORBIT_RADIUS - ARRIVAL_START_RADIUS) * s
    return (float(STAR_POS[0] + radius), float(STAR_POS[1]))


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
            tug_trail=deque(maxlen=MAX_TRAIL_POINTS),
            pilot_age=0.0,
            stranded_angle=stranded_angle,
            stranded_pos=circular_orbit_position(
                STAR_POS, STRANDED_ORBIT_RADIUS, stranded_angle
            ),
            arrival_progress=0.0 if with_arrival else 1.0,
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
        vel = circular_orbit_velocity(STRANDED_ORBIT_RADIUS, self.stranded_angle)
        self.tug_pos = self.stranded_pos
        self.tug_vel = vel
        self.tug_angle = math.atan2(vel[1], vel[0])
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
    if gs.state in (State.WON, State.FAILED):
        return

    if gs.state is State.ARRIVING:
        _advance_stranded(gs, dt, stranded_omega)
        gs.arrival_progress += dt / ARRIVAL_DURATION
        if gs.arrival_progress >= 1.0:
            gs.finish_arrival()
        return

    if gs.state in (State.PARKED, State.OUTBOUND, State.DOCKED):
        _advance_stranded(gs, dt, stranded_omega)

    if gs.state not in (State.OUTBOUND, State.HOMEBOUND):
        return

    if inp.left:
        gs.tug_angle -= TUG_ROT_SPEED * dt
    if inp.right:
        gs.tug_angle += TUG_ROT_SPEED * dt

    thrust = (
        (math.cos(gs.tug_angle) * TUG_THRUST, math.sin(gs.tug_angle) * TUG_THRUST)
        if inp.thrust
        else (0.0, 0.0)
    )
    gs.tug_pos, gs.tug_vel = step(gs.tug_pos, gs.tug_vel, STAR_POS, dt, thrust)
    gs.tug_trail.append(gs.tug_pos)
    gs.pilot_age += dt

    if gs.state is State.HOMEBOUND:
        gs.stranded_pos = gs.tug_pos

    outbound = gs.state is State.OUTBOUND

    pdx = gs.tug_pos[0] - STAR_POS[0]
    pdy = gs.tug_pos[1] - STAR_POS[1]
    if pdx * pdx + pdy * pdy <= STAR_RADIUS * STAR_RADIUS:
        gs.status = (
            "Crashed the tug into the star. (R to reset)"
            if outbound
            else "Crashed on the way home. (R to reset)"
        )
        gs.state = State.FAILED
        return

    if outbound:
        dx = gs.tug_pos[0] - gs.stranded_pos[0]
        dy = gs.tug_pos[1] - gs.stranded_pos[1]
        if dx * dx + dy * dy <= capture_radius * capture_radius:
            gs.state = State.DOCKED
            gs.status = "Docked! SPACE to engage the tug."
            return
    else:
        rdx = gs.tug_pos[0] - RESCUE_POS[0]
        rdy = gs.tug_pos[1] - RESCUE_POS[1]
        if rdx * rdx + rdy * rdy <= DOCK_RADIUS * DOCK_RADIUS:
            gs.tug_pos = (float(RESCUE_POS[0]), float(RESCUE_POS[1]))
            gs.stranded_pos = gs.tug_pos
            gs.state = State.WON
            gs.status = "Rescue complete!"
            return

    if off_screen(gs.tug_pos) or gs.pilot_age >= MAX_PILOT_SECONDS:
        gs.status = (
            "Tug lost in space. (R to reset)"
            if outbound
            else "Lost on the way home. (R to reset)"
        )
        gs.state = State.FAILED


def _advance_stranded(gs: GameState, dt: float, omega: float) -> None:
    gs.stranded_angle = (gs.stranded_angle + omega * dt) % (2 * math.pi)
    gs.stranded_pos = circular_orbit_position(
        STAR_POS, STRANDED_ORBIT_RADIUS, gs.stranded_angle
    )


def _settings_lines(
    capture_setting: str, stellar_damage: bool
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    return [
        [("[c]", HUD_ACTION), (f"apture radius: {capture_setting}", HUD)],
        [("[d]", HUD_ACTION), (f"amage: {'on' if stellar_damage else 'off'}", HUD)],
    ]


def _center_content(
    gs: GameState, briefing_visible: bool
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    if briefing_visible:
        return []
    if gs.state is State.ARRIVING:
        return [[("[SPACE]", HUD_ACTION), (" skip intro", HUD)]]
    if gs.state is State.PARKED:
        return [[("[SPACE]", HUD_ACTION), (" launch tug", HUD)]]
    if gs.state is State.DOCKED:
        return [
            [("docked with ", HUD), ("stranded", STRANDED), (" vessel", HUD)],
            [("[SPACE]", HUD_ACTION), (" to begin return journey", HUD)],
        ]
    color = (
        HUD_OK if gs.state is State.WON
        else HUD_BAD if gs.state is State.FAILED
        else HUD
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

    # TOP-LEFT: distance from star
    r = int(math.hypot(gs.tug_pos[0] - STAR_POS[0], gs.tug_pos[1] - STAR_POS[1]))
    screen.blit(font.render(f"R: {r} px", True, HUD), (pad, pad))

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

    # Tug parked inside the rescue ship until it launches.
    if gs.state in (State.ARRIVING, State.PARKED):
        dx = STAR_POS[0] - rescue_pos[0]
        dy = STAR_POS[1] - rescue_pos[1]
        rescue_facing_out = math.atan2(dy, dx) + math.pi
        render.draw_capture_halo(screen, rescue_pos, capture_radius)
        render.draw_tug(screen, rescue_pos, rescue_facing_out, thrusting=False, cargo=False)

    if gs.state in (State.ARRIVING, State.PARKED, State.OUTBOUND):
        render.draw_stranded(screen, gs.stranded_pos)

    if gs.state is State.OUTBOUND:
        render.draw_capture_halo(screen, gs.tug_pos, capture_radius)
        render.draw_tug(screen, gs.tug_pos, gs.tug_angle, thrusting, cargo=False)
    elif gs.state is State.DOCKED:
        tug_c = render.tug_visual_center(gs.stranded_pos, gs.tug_angle, cargo=True)
        render.draw_capture_halo(screen, tug_c, capture_radius)
        render.draw_tug(screen, gs.stranded_pos, gs.tug_angle, thrusting=False, cargo=True)
    elif gs.state in (State.HOMEBOUND, State.WON, State.FAILED):
        tug_c = render.tug_visual_center(gs.tug_pos, gs.tug_angle, cargo=True)
        render.draw_capture_halo(screen, tug_c, capture_radius)
        render.draw_tug(screen, gs.tug_pos, gs.tug_angle, thrusting, cargo=True)

    _draw_hud(screen, font, gs, capture_setting, stellar_damage, briefing_visible)

    if briefing_visible:
        render.draw_briefing_modal(
            screen,
            title_font,
            font,
            BRIEFING_TITLE,
            BRIEFING_LINES,
            [("[SPACE]", HUD_ACTION), (" to continue", HUD)],
        )


async def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Orbital Rescue")
    clock = pygame.time.Clock()
    font_path = str(Path(__file__).parent / "Moby-Monospace.ttf")
    font = pygame.font.Font(font_path, 18)
    title_font = pygame.font.Font(font_path, 28)

    stranded_omega = circular_orbit_angular_speed(STRANDED_ORBIT_RADIUS)
    capture_setting = "medium"
    stellar_damage = False
    briefing_dismissed = False
    gs = GameState.new(with_arrival=True)
    accumulator = 0.0

    running = True
    while running:
        real_dt = clock.tick(FPS) / 1000.0
        accumulator += min(real_dt, MAX_FRAME_DT)

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
                    i = CAPTURE_CYCLE.index(capture_setting)
                    capture_setting = CAPTURE_CYCLE[(i + 1) % len(CAPTURE_CYCLE)]
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
                        gs.engage_homebound()

        keys = pygame.key.get_pressed()
        piloting = gs.state in (State.OUTBOUND, State.HOMEBOUND)
        inp = InputFrame(
            left=piloting and keys[pygame.K_LEFT],
            right=piloting and keys[pygame.K_RIGHT],
            thrust=piloting and keys[pygame.K_UP],
        )
        capture_radius = CAPTURE_RADII[capture_setting]

        while accumulator >= DT_SIM:
            simulate(gs, DT_SIM, inp, capture_radius, stranded_omega)
            accumulator -= DT_SIM

        briefing_visible = gs.state is State.PARKED and not briefing_dismissed
        draw(screen, font, title_font, gs, inp.thrust, capture_radius, capture_setting, stellar_damage, briefing_visible)
        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
