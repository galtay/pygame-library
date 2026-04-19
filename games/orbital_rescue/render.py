"""Drawing primitives and HUD for Orbital Rescue."""

from __future__ import annotations

import math
import random
from typing import Iterable

import pygame

import constants


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


def draw_background(surf: pygame.Surface, dim_star: bool = False) -> None:
    surf.fill(constants.BG)
    pygame.draw.circle(surf, constants.ORBIT_GUIDE, constants.STAR_POS, constants.STRANDED_ORBIT_RADIUS, 1)
    pygame.draw.circle(surf, constants.ORBIT_GUIDE, constants.STAR_POS, constants.RESCUE_ORBIT_RADIUS, 1)
    # Star: concentric opaque circles from dim outer glow to hot core.
    # `dim_star` drops each layer to ~33% brightness so mission briefing text
    # stays legible over the star.
    k = 0.33 if dim_star else 1.0
    layers = (
        (constants.STAR_OUTER, constants.STAR_OUTER_RADIUS),
        (constants.STAR_GLOW, constants.STAR_GLOW_RADIUS),
        (constants.STAR_CORONA, constants.STAR_CORONA_RADIUS),
        (constants.STAR_CORE, constants.STAR_RADIUS),
    )
    for color, radius in layers:
        dim = (int(color[0] * k), int(color[1] * k), int(color[2] * k))
        pygame.draw.circle(surf, dim, constants.STAR_POS, radius)


def draw_trail(surf: pygame.Surface, trail: Iterable[tuple[float, float]]) -> None:
    points = list(trail)
    n = len(points)
    if n < 2:
        return
    bg_r, bg_g, bg_b = constants.BG
    tr_r, tr_g, tr_b = constants.TRAIL
    dr, dg, db = tr_r - bg_r, tr_g - bg_g, tr_b - bg_b
    denom = n - 1
    for i in range(denom):
        t = (i + 1) / denom
        color = (
            int(bg_r + dr * t),
            int(bg_g + dg * t),
            int(bg_b + db * t),
        )
        pygame.draw.line(surf, color, points[i], points[i + 1], 1)


def draw_rescue_ship(surf: pygame.Surface, pos: tuple[float, float]) -> None:
    dx = constants.STAR_POS[0] - pos[0]
    dy = constants.STAR_POS[1] - pos[1]
    d = math.hypot(dx, dy)
    if d == 0.0:
        return
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    facing_out = math.atan2(uy, ux) + math.pi
    draw_triangle(surf, constants.RESCUE, pos, facing_out, constants.RESCUE_SIZE)
    for offset in (-5, 5):
        base = (
            pos[0] + ux * 10 + px * offset,
            pos[1] + uy * 10 + py * offset,
        )
        flame_len = 14 + random.uniform(-2, 4)
        tip = (base[0] + ux * flame_len, base[1] + uy * flame_len)
        pygame.draw.line(surf, constants.FLAME, base, tip, 2)


def draw_stranded(
    surf: pygame.Surface,
    pos: tuple[float, float],
    facing: float = constants.STRANDED_FACING,
) -> None:
    draw_triangle(surf, constants.STRANDED, pos, facing, constants.STRANDED_SIZE)


CARGO_TUG_OFFSET = 12.0


def tug_visual_center(
    craft_pos: tuple[float, float], facing: float, cargo: bool
) -> tuple[float, float]:
    """Where the tug triangle will actually be drawn (offset backward when towing cargo)."""
    if not cargo:
        return (float(craft_pos[0]), float(craft_pos[1]))
    return (
        craft_pos[0] - math.cos(facing) * CARGO_TUG_OFFSET,
        craft_pos[1] - math.sin(facing) * CARGO_TUG_OFFSET,
    )


def draw_tug(
    surf: pygame.Surface,
    pos: tuple[float, float],
    facing: float,
    thrusting: bool,
    cargo: bool,
) -> None:
    if cargo:
        draw_triangle(surf, constants.STRANDED, pos, facing, constants.STRANDED_SIZE)
    tug_pos = tug_visual_center(pos, facing, cargo)
    draw_triangle(surf, constants.TUG, tug_pos, facing, constants.TUG_SIZE)
    if thrusting:
        back = facing + math.pi
        base = (tug_pos[0] + math.cos(back) * 8, tug_pos[1] + math.sin(back) * 8)
        tip = (tug_pos[0] + math.cos(back) * 20, tug_pos[1] + math.sin(back) * 20)
        pygame.draw.line(surf, constants.FLAME, base, tip, 2)


def draw_capture_halo(
    surf: pygame.Surface, pos: tuple[float, float], radius: float
) -> None:
    pygame.draw.circle(surf, constants.HALO, (int(pos[0]), int(pos[1])), int(radius), 1)


def render_key_line(
    font: pygame.font.Font,
    segments: list[tuple[str, tuple[int, int, int]]],
) -> pygame.Surface:
    """Combine colored text runs into a single left-to-right surface."""
    if not segments:
        return pygame.Surface((0, font.get_height()), pygame.SRCALPHA)
    rendered = [font.render(text, True, color) for text, color in segments]
    width = sum(s.get_width() for s in rendered)
    height = max(s.get_height() for s in rendered)
    combined = pygame.Surface((width, height), pygame.SRCALPHA)
    x = 0
    for s in rendered:
        combined.blit(s, (x, 0))
        x += s.get_width()
    return combined


def draw_briefing_modal(
    surf: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    title: str,
    body_lines: list[list[tuple[str, tuple[int, int, int]]]],
    prompt_segments: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Centered mission briefing: title, segmented body lines, and a key-highlighted prompt."""
    screen_w, screen_h = surf.get_size()

    title_surf = title_font.render(title, True, constants.HUD_ACTION)
    body_surfs = [render_key_line(body_font, line) for line in body_lines]
    prompt_surf = render_key_line(body_font, prompt_segments)

    title_gap = 18
    body_line_h = body_font.get_height() + 4
    prompt_gap = 24

    total_h = (
        title_surf.get_height()
        + title_gap
        + body_line_h * len(body_lines)
        + prompt_gap
        + prompt_surf.get_height()
    )
    y = (screen_h - total_h) // 2
    cx = screen_w // 2

    surf.blit(title_surf, (cx - title_surf.get_width() // 2, y))
    y += title_surf.get_height() + title_gap

    for s in body_surfs:
        if s.get_width() > 0:
            surf.blit(s, (cx - s.get_width() // 2, y))
        y += body_line_h

    y += prompt_gap
    surf.blit(prompt_surf, (cx - prompt_surf.get_width() // 2, y))
