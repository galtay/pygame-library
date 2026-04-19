"""Orbital Rescue — pilot a tug from the rescue ship to a stranded ship in
circular orbit around a planet, then pilot the combined craft home.

Three phases:

1. OUTBOUND  — tug starts in a circular orbit at the rescue ship's radius.
               Player rotates + thrusts to intercept the stranded ship.
2. DOCKED    — combined body rides the stranded ship's circular orbit.
               Player chooses when to engage the tug controls.
3. HOMEBOUND — player pilots the tug + cargo back to the rescue ship's
               fixed position, fighting gravity.
"""

import math
import random
import sys
from enum import Enum, auto

import pygame

from physics import (
    circular_orbit_angular_speed,
    circular_orbit_position,
    circular_orbit_velocity,
    gravity_accel,
    step,
)

WINDOW_SIZE = (900, 700)
FPS = 60
DT = 1.0
PHYSICS_SUBSTEPS = 4

BG = (5, 5, 15)
PLANET_CORE = (40, 70, 110)
PLANET_RIM = (140, 180, 230)
STRANDED = (240, 220, 120)
RESCUE = (130, 230, 160)
TUG = (130, 220, 255)
FLAME = (255, 170, 80)
TRAIL = (120, 120, 150)
ORBIT_GUIDE = (35, 35, 55)
HALO = (110, 200, 140)
DOCK_LINK = (200, 240, 180)
HUD = (180, 180, 200)
HUD_OK = (170, 240, 180)
HUD_BAD = (240, 150, 150)
HUD_ACTION = (240, 220, 140)

PLANET_POS = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
PLANET_RADIUS = 32
STRANDED_ORBIT_RADIUS = 220
RESCUE_ORBIT_RADIUS = 320
RESCUE_POS = (PLANET_POS[0] + RESCUE_ORBIT_RADIUS, PLANET_POS[1])

MAX_TRAIL_LEN = 1200
MAX_PILOT_FRAMES = 60 * 180

DIFFICULTIES: dict[str, float] = {
    "easy": 30.0,
    "normal": 15.0,
    "hard": 7.0,
}
DIFFICULTY_KEYS = {
    pygame.K_1: "easy",
    pygame.K_2: "normal",
    pygame.K_3: "hard",
}

TUG_THRUST = 0.08
TUG_ROT_STEP = math.radians(3.0)
DOCK_RADIUS = 18.0


class State(Enum):
    PARKED = auto()
    OUTBOUND = auto()
    DOCKED = auto()
    HOMEBOUND = auto()
    WON = auto()
    FAILED = auto()


def draw_triangle(
    surf: pygame.Surface,
    color: tuple[int, int, int],
    pos: tuple[float, float],
    facing: float,
    size: float,
    width: int = 1,
) -> None:
    cx, cy = pos
    tip = (cx + math.cos(facing) * size, cy + math.sin(facing) * size)
    left = (
        cx + math.cos(facing + 2.5) * size * 0.6,
        cy + math.sin(facing + 2.5) * size * 0.6,
    )
    right = (
        cx + math.cos(facing - 2.5) * size * 0.6,
        cy + math.sin(facing - 2.5) * size * 0.6,
    )
    pygame.draw.polygon(surf, color, [tip, left, right], width)


def draw_rescue_ship(surf: pygame.Surface) -> None:
    dx = PLANET_POS[0] - RESCUE_POS[0]
    dy = PLANET_POS[1] - RESCUE_POS[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    facing_out = math.atan2(-uy, -ux)
    draw_triangle(surf, RESCUE, RESCUE_POS, facing_out, 18)
    for offset in (-5, 5):
        base = (
            RESCUE_POS[0] + ux * 10 + px * offset,
            RESCUE_POS[1] + uy * 10 + py * offset,
        )
        flame_len = 14 + random.uniform(-2, 4)
        tip = (base[0] + ux * flame_len, base[1] + uy * flame_len)
        pygame.draw.line(surf, FLAME, base, tip, 2)


def draw_tug(
    surf: pygame.Surface,
    pos: tuple[float, float],
    facing: float,
    thrusting: bool,
    cargo: bool,
) -> None:
    draw_triangle(surf, TUG, pos, facing, 13)
    if cargo:
        draw_triangle(surf, STRANDED, pos, facing, 6)
    if thrusting:
        back = facing + math.pi
        base = (pos[0] + math.cos(back) * 8, pos[1] + math.sin(back) * 8)
        tip = (pos[0] + math.cos(back) * 20, pos[1] + math.sin(back) * 20)
        pygame.draw.line(surf, FLAME, base, tip, 2)


def integrate_body(
    pos: tuple[float, float],
    vel: tuple[float, float],
    dt: float,
    thrust: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if thrust is None:
        return step(pos, vel, PLANET_POS, dt)
    gax, gay = gravity_accel(pos, PLANET_POS)
    tax, tay = thrust
    vx = vel[0] + (gax + tax) * dt
    vy = vel[1] + (gay + tay) * dt
    px = pos[0] + vx * dt
    py = pos[1] + vy * dt
    return (px, py), (vx, vy)


def off_screen(pos: tuple[float, float], margin: float = 400.0) -> bool:
    return (
        pos[0] < -margin
        or pos[0] > WINDOW_SIZE[0] + margin
        or pos[1] < -margin
        or pos[1] > WINDOW_SIZE[1] + margin
    )


def initial_tug_state() -> tuple[tuple[float, float], tuple[float, float], float]:
    """Tug launched from the rescue ship into a circular orbit at r=320."""
    pos = (float(RESCUE_POS[0]), float(RESCUE_POS[1]))
    vel = circular_orbit_velocity(RESCUE_ORBIT_RADIUS, 0.0)
    facing = math.atan2(vel[1], vel[0])
    return pos, vel, facing


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Orbital Rescue")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    stranded_omega = circular_orbit_angular_speed(STRANDED_ORBIT_RADIUS)
    stranded_angle = random.uniform(0.0, 2 * math.pi)

    difficulty = "normal"

    state = State.PARKED
    tug_pos, _, tug_angle = initial_tug_state()
    tug_vel: tuple[float, float] = (0.0, 0.0)
    tug_trail: list[tuple[float, float]] = []
    pilot_age = 0
    stranded_pos: tuple[float, float] = circular_orbit_position(
        PLANET_POS, STRANDED_ORBIT_RADIUS, stranded_angle
    )
    status = "SPACE to launch the tug."

    def reset() -> None:
        nonlocal state, tug_pos, tug_vel, tug_angle, tug_trail, pilot_age
        nonlocal stranded_angle, status
        state = State.PARKED
        tug_pos, _, tug_angle = initial_tug_state()
        tug_vel = (0.0, 0.0)
        tug_trail = []
        pilot_age = 0
        stranded_angle = random.uniform(0.0, 2 * math.pi)
        status = "SPACE to launch the tug."

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset()
                elif event.key in DIFFICULTY_KEYS and state in (State.PARKED, State.OUTBOUND):
                    difficulty = DIFFICULTY_KEYS[event.key]
                elif event.key == pygame.K_SPACE and state is State.PARKED:
                    tug_pos, tug_vel, tug_angle = initial_tug_state()
                    tug_trail = [tug_pos]
                    pilot_age = 0
                    state = State.OUTBOUND
                    status = "Pilot the tug to the stranded ship."
                elif event.key == pygame.K_SPACE and state is State.DOCKED:
                    tug_pos = stranded_pos
                    tug_vel = circular_orbit_velocity(
                        STRANDED_ORBIT_RADIUS, stranded_angle
                    )
                    tug_angle = math.atan2(tug_vel[1], tug_vel[0])
                    tug_trail = [tug_pos]
                    pilot_age = 0
                    state = State.HOMEBOUND
                    status = "Piloting home with cargo."

        keys = pygame.key.get_pressed()
        thrusting = False
        if state in (State.OUTBOUND, State.HOMEBOUND):
            if keys[pygame.K_LEFT]:
                tug_angle -= TUG_ROT_STEP
            if keys[pygame.K_RIGHT]:
                tug_angle += TUG_ROT_STEP
            if keys[pygame.K_UP]:
                thrusting = True

        if state in (State.PARKED, State.OUTBOUND, State.DOCKED):
            stranded_angle = (stranded_angle + stranded_omega * DT) % (2 * math.pi)
            stranded_pos = circular_orbit_position(
                PLANET_POS, STRANDED_ORBIT_RADIUS, stranded_angle
            )
        elif state is State.HOMEBOUND:
            stranded_pos = tug_pos

        capture_radius = DIFFICULTIES[difficulty]

        if state in (State.OUTBOUND, State.HOMEBOUND):
            sub_dt = DT / PHYSICS_SUBSTEPS
            thrust = (
                (math.cos(tug_angle) * TUG_THRUST, math.sin(tug_angle) * TUG_THRUST)
                if thrusting
                else (0.0, 0.0)
            )
            for _ in range(PHYSICS_SUBSTEPS):
                tug_pos, tug_vel = integrate_body(tug_pos, tug_vel, sub_dt, thrust)
                pdx = tug_pos[0] - PLANET_POS[0]
                pdy = tug_pos[1] - PLANET_POS[1]
                if pdx * pdx + pdy * pdy <= PLANET_RADIUS * PLANET_RADIUS:
                    status = (
                        "Crashed the tug into the planet. (R to reset)"
                        if state is State.OUTBOUND
                        else "Crashed on the way home. (R to reset)"
                    )
                    state = State.FAILED
                    break
                if state is State.OUTBOUND:
                    dx = tug_pos[0] - stranded_pos[0]
                    dy = tug_pos[1] - stranded_pos[1]
                    if dx * dx + dy * dy <= capture_radius * capture_radius:
                        state = State.DOCKED
                        status = "Docked! SPACE to engage the tug."
                        break
                else:
                    rdx = tug_pos[0] - RESCUE_POS[0]
                    rdy = tug_pos[1] - RESCUE_POS[1]
                    if rdx * rdx + rdy * rdy <= DOCK_RADIUS * DOCK_RADIUS:
                        tug_pos = (float(RESCUE_POS[0]), float(RESCUE_POS[1]))
                        stranded_pos = tug_pos
                        state = State.WON
                        status = "Rescue complete! (R to reset)"
                        break
            tug_trail.append(tug_pos)
            if len(tug_trail) > MAX_TRAIL_LEN:
                tug_trail.pop(0)
            pilot_age += 1
            if state in (State.OUTBOUND, State.HOMEBOUND):
                if off_screen(tug_pos) or pilot_age >= MAX_PILOT_FRAMES:
                    status = (
                        "Tug lost in space. (R to reset)"
                        if state is State.OUTBOUND
                        else "Lost on the way home. (R to reset)"
                    )
                    state = State.FAILED

        screen.fill(BG)
        pygame.draw.circle(screen, ORBIT_GUIDE, PLANET_POS, STRANDED_ORBIT_RADIUS, 1)
        pygame.draw.circle(screen, ORBIT_GUIDE, PLANET_POS, RESCUE_ORBIT_RADIUS, 1)
        pygame.draw.circle(screen, PLANET_CORE, PLANET_POS, PLANET_RADIUS)
        pygame.draw.circle(screen, PLANET_RIM, PLANET_POS, PLANET_RADIUS, 1)

        if len(tug_trail) >= 2:
            pygame.draw.lines(screen, TRAIL, False, tug_trail, 1)

        draw_rescue_ship(screen)

        if state in (State.PARKED, State.OUTBOUND):
            s_facing = math.atan2(
                stranded_pos[1] - PLANET_POS[1],
                stranded_pos[0] - PLANET_POS[0],
            ) + math.pi / 2
            draw_triangle(screen, STRANDED, stranded_pos, s_facing, 9)

        if state is State.OUTBOUND:
            ipos = (int(tug_pos[0]), int(tug_pos[1]))
            pygame.draw.circle(screen, HALO, ipos, int(capture_radius), 1)
            draw_tug(screen, tug_pos, tug_angle, thrusting, cargo=False)

        elif state is State.DOCKED:
            prograde = stranded_angle + math.pi / 2
            draw_tug(screen, stranded_pos, prograde, thrusting=False, cargo=True)
            pygame.draw.circle(
                screen, DOCK_LINK, (int(stranded_pos[0]), int(stranded_pos[1])), 16, 1
            )

        elif state in (State.HOMEBOUND, State.WON, State.FAILED):
            draw_tug(screen, tug_pos, tug_angle, thrusting, cargo=True)

        status_color = (
            HUD_OK if state is State.WON
            else HUD_BAD if state is State.FAILED
            else HUD_ACTION if state in (State.PARKED, State.DOCKED)
            else HUD
        )
        if state is State.PARKED:
            line2 = "SPACE launch tug   1/2/3 difficulty   R reset   ESC quit"
        elif state is State.OUTBOUND:
            line2 = "LEFT/RIGHT rotate   UP thrust   R reset   ESC quit"
        elif state is State.DOCKED:
            line2 = "SPACE engage tug   R reset   ESC quit"
        elif state is State.HOMEBOUND:
            line2 = "LEFT/RIGHT rotate   UP thrust   R reset   ESC quit"
        else:
            line2 = "R reset   ESC quit"

        hud = [
            (f"difficulty: {difficulty}  (capture r={int(capture_radius)})    "
             f"tug v: {math.hypot(tug_vel[0], tug_vel[1]):4.2f} px/f", HUD),
            (line2, HUD),
            (status, status_color),
        ]
        for i, (line, color) in enumerate(hud):
            screen.blit(font.render(line, True, color), (12, 10 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()
    sys.exit(0)
