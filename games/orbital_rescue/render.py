"""Drawing primitives and HUD for Orbital Rescue."""

from __future__ import annotations

import math
import random
from typing import Iterable

import pygame

from constants import (
    BG,
    DOCK_LINK,
    FLAME,
    HALO,
    ORBIT_GUIDE,
    PLANET_CORE,
    PLANET_POS,
    PLANET_RADIUS,
    PLANET_RIM,
    RESCUE,
    RESCUE_ORBIT_RADIUS,
    RESCUE_POS,
    STRANDED,
    STRANDED_ORBIT_RADIUS,
    TRAIL,
    TUG,
)


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


def draw_background(surf: pygame.Surface) -> None:
    surf.fill(BG)
    pygame.draw.circle(surf, ORBIT_GUIDE, PLANET_POS, STRANDED_ORBIT_RADIUS, 1)
    pygame.draw.circle(surf, ORBIT_GUIDE, PLANET_POS, RESCUE_ORBIT_RADIUS, 1)
    pygame.draw.circle(surf, PLANET_CORE, PLANET_POS, PLANET_RADIUS)
    pygame.draw.circle(surf, PLANET_RIM, PLANET_POS, PLANET_RADIUS, 1)


def draw_trail(surf: pygame.Surface, trail: Iterable[tuple[float, float]]) -> None:
    points = list(trail)
    if len(points) >= 2:
        pygame.draw.lines(surf, TRAIL, False, points, 1)


def draw_rescue_ship(surf: pygame.Surface) -> None:
    dx = PLANET_POS[0] - RESCUE_POS[0]
    dy = PLANET_POS[1] - RESCUE_POS[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    facing_out = math.atan2(uy, ux) + math.pi
    draw_triangle(surf, RESCUE, RESCUE_POS, facing_out, 18)
    for offset in (-5, 5):
        base = (
            RESCUE_POS[0] + ux * 10 + px * offset,
            RESCUE_POS[1] + uy * 10 + py * offset,
        )
        flame_len = 14 + random.uniform(-2, 4)
        tip = (base[0] + ux * flame_len, base[1] + uy * flame_len)
        pygame.draw.line(surf, FLAME, base, tip, 2)


def draw_stranded(surf: pygame.Surface, pos: tuple[float, float]) -> None:
    facing = (
        math.atan2(pos[1] - PLANET_POS[1], pos[0] - PLANET_POS[0]) + math.pi / 2
    )
    draw_triangle(surf, STRANDED, pos, facing, 9)


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


def draw_capture_halo(
    surf: pygame.Surface, pos: tuple[float, float], radius: float
) -> None:
    pygame.draw.circle(surf, HALO, (int(pos[0]), int(pos[1])), int(radius), 1)


def draw_dock_link(surf: pygame.Surface, pos: tuple[float, float]) -> None:
    pygame.draw.circle(surf, DOCK_LINK, (int(pos[0]), int(pos[1])), 16, 1)


def draw_hud(
    surf: pygame.Surface,
    font: pygame.font.Font,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
    for i, (line, color) in enumerate(lines):
        surf.blit(font.render(line, True, color), (12, 10 + i * 18))
