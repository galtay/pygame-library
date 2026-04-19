"""Game loop, state machine, and input handling for Orbital Rescue.

The loop runs a fixed-timestep simulation driven by real elapsed time, so
gameplay is identical regardless of the display refresh rate. Rendering
runs once per loop iteration at whatever rate the platform delivers.

The loop is async so the same code runs under pygbag in the browser; the
`await asyncio.sleep(0)` after each flip yields to the JS event loop.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import pygame

from constants import (
    DIFFICULTIES,
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
    PLANET_POS,
    PLANET_RADIUS,
    RESCUE_ORBIT_RADIUS,
    RESCUE_POS,
    STRANDED_ORBIT_RADIUS,
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
    status: str

    @classmethod
    def new(cls) -> "GameState":
        """Fresh game: tug parked on the rescue ship, stranded ship at a random phase."""
        pos, _vel, facing = tug_launch_orbit()
        stranded_angle = random_stranded_angle()
        return cls(
            state=State.PARKED,
            tug_pos=pos,
            tug_vel=(0.0, 0.0),
            tug_angle=facing,
            tug_trail=deque(maxlen=MAX_TRAIL_POINTS),
            pilot_age=0.0,
            stranded_angle=stranded_angle,
            stranded_pos=circular_orbit_position(
                PLANET_POS, STRANDED_ORBIT_RADIUS, stranded_angle
            ),
            status="SPACE to launch the tug.",
        )

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
    gs.tug_pos, gs.tug_vel = step(gs.tug_pos, gs.tug_vel, PLANET_POS, dt, thrust)
    gs.tug_trail.append(gs.tug_pos)
    gs.pilot_age += dt

    if gs.state is State.HOMEBOUND:
        gs.stranded_pos = gs.tug_pos

    outbound = gs.state is State.OUTBOUND

    pdx = gs.tug_pos[0] - PLANET_POS[0]
    pdy = gs.tug_pos[1] - PLANET_POS[1]
    if pdx * pdx + pdy * pdy <= PLANET_RADIUS * PLANET_RADIUS:
        gs.status = (
            "Crashed the tug into the planet. (R to reset)"
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
            gs.status = "Rescue complete! (R to reset)"
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
        PLANET_POS, STRANDED_ORBIT_RADIUS, gs.stranded_angle
    )


def draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    gs: GameState,
    thrusting: bool,
    capture_radius: float,
    difficulty: str,
) -> None:
    render.draw_background(screen)
    render.draw_trail(screen, gs.tug_trail)
    render.draw_rescue_ship(screen)

    if gs.state in (State.PARKED, State.OUTBOUND):
        render.draw_stranded(screen, gs.stranded_pos)

    if gs.state is State.OUTBOUND:
        render.draw_capture_halo(screen, gs.tug_pos, capture_radius)
        render.draw_tug(screen, gs.tug_pos, gs.tug_angle, thrusting, cargo=False)
    elif gs.state is State.DOCKED:
        prograde = gs.stranded_angle + math.pi / 2
        render.draw_tug(screen, gs.stranded_pos, prograde, thrusting=False, cargo=True)
        render.draw_dock_link(screen, gs.stranded_pos)
    elif gs.state in (State.HOMEBOUND, State.WON, State.FAILED):
        render.draw_tug(screen, gs.tug_pos, gs.tug_angle, thrusting, cargo=True)

    status_color = (
        HUD_OK if gs.state is State.WON
        else HUD_BAD if gs.state is State.FAILED
        else HUD_ACTION if gs.state in (State.PARKED, State.DOCKED)
        else HUD
    )
    if gs.state is State.PARKED:
        line2 = "SPACE launch tug   1/2/3 difficulty   R reset   ESC quit"
    elif gs.state in (State.OUTBOUND, State.HOMEBOUND):
        line2 = "LEFT/RIGHT rotate   UP thrust   R reset   ESC quit"
    elif gs.state is State.DOCKED:
        line2 = "SPACE engage tug   R reset   ESC quit"
    else:
        line2 = "R reset   ESC quit"

    render.draw_hud(
        screen,
        font,
        [
            (
                f"difficulty: {difficulty}  (capture r={int(capture_radius)})    "
                f"tug v: {math.hypot(gs.tug_vel[0], gs.tug_vel[1]):6.1f} px/s",
                HUD,
            ),
            (line2, HUD),
            (gs.status, status_color),
        ],
    )


async def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Orbital Rescue")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 18)

    stranded_omega = circular_orbit_angular_speed(STRANDED_ORBIT_RADIUS)
    difficulty_keys = {
        pygame.K_1: "easy",
        pygame.K_2: "normal",
        pygame.K_3: "hard",
    }
    difficulty = "normal"
    gs = GameState.new()
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
                    accumulator = 0.0
                elif event.key in difficulty_keys and gs.state in (
                    State.PARKED,
                    State.OUTBOUND,
                ):
                    difficulty = difficulty_keys[event.key]
                elif event.key == pygame.K_SPACE and gs.state is State.PARKED:
                    gs.launch_outbound()
                elif event.key == pygame.K_SPACE and gs.state is State.DOCKED:
                    gs.engage_homebound()

        keys = pygame.key.get_pressed()
        piloting = gs.state in (State.OUTBOUND, State.HOMEBOUND)
        inp = InputFrame(
            left=piloting and keys[pygame.K_LEFT],
            right=piloting and keys[pygame.K_RIGHT],
            thrust=piloting and keys[pygame.K_UP],
        )
        capture_radius = DIFFICULTIES[difficulty]

        while accumulator >= DT_SIM:
            simulate(gs, DT_SIM, inp, capture_radius, stranded_omega)
            accumulator -= DT_SIM

        draw(screen, font, gs, inp.thrust, capture_radius, difficulty)
        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
