"""Drawing primitives and HUD for Orbital Rescue."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import pygame

import constants
from geometry import tug_visual_center


@dataclass(frozen=True)
class Camera:
    """Star-anchored view. World point STAR_POS always renders at
    VIEWPORT_CENTER on screen, regardless of zoom; other world points contract
    toward the star as zoom drops.
    """

    zoom: float = 1.0

    def to_screen(self, pos: tuple[float, float]) -> tuple[float, float]:
        wx, wy = constants.STAR_POS
        sx, sy = constants.VIEWPORT_CENTER
        return (sx + (pos[0] - wx) * self.zoom, sy + (pos[1] - wy) * self.zoom)

    def s(self, length: float) -> float:
        return length * self.zoom


def draw_triangle(
    surf: pygame.Surface,
    color: tuple[int, int, int],
    pos: tuple[float, float],
    facing: float,
    size: float,
    cam: Camera,
    width: int = 1,
) -> None:
    cx, cy = cam.to_screen(pos)
    s = cam.s(size)
    tip = (cx + math.cos(facing) * s, cy + math.sin(facing) * s)
    left = (
        cx + math.cos(facing + 2.5) * s * 0.6,
        cy + math.sin(facing + 2.5) * s * 0.6,
    )
    right = (
        cx + math.cos(facing - 2.5) * s * 0.6,
        cy + math.sin(facing - 2.5) * s * 0.6,
    )
    pygame.draw.polygon(surf, color, [tip, left, right], width)


def draw_background(surf: pygame.Surface, cam: Camera, dim_star: bool = False) -> None:
    surf.fill(constants.BG)
    star_screen = cam.to_screen(constants.STAR_POS)
    pygame.draw.circle(surf, constants.ORBIT_GUIDE, star_screen, cam.s(constants.STRANDED_ORBIT_RADIUS), 1)
    pygame.draw.circle(surf, constants.ORBIT_GUIDE, star_screen, cam.s(constants.RESCUE_ORBIT_RADIUS), 1)
    # Lost-in-space boundary: three concentric red rings (faint outer halo,
    # mid glow, bright core) for a glowing-edge effect. Ring offsets are in
    # screen pixels so the glow stays visible at every zoom.
    lost_r = int(cam.s(constants.LOST_RADIUS))
    if lost_r > 0:
        pygame.draw.circle(surf, constants.LOST_BOUNDARY_HALO, star_screen, lost_r + 2, 1)
        pygame.draw.circle(surf, constants.LOST_BOUNDARY_GLOW, star_screen, lost_r + 1, 1)
        pygame.draw.circle(surf, constants.LOST_BOUNDARY, star_screen, lost_r, 1)
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
        pygame.draw.circle(surf, dim, star_screen, cam.s(radius))


def draw_trail(
    surf: pygame.Surface, trail: Iterable[tuple[float, float]], cam: Camera
) -> None:
    points = [cam.to_screen(p) for p in trail]
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


def draw_rescue_ship(surf: pygame.Surface, pos: tuple[float, float], cam: Camera) -> None:
    dx = constants.STAR_POS[0] - pos[0]
    dy = constants.STAR_POS[1] - pos[1]
    d = math.hypot(dx, dy)
    if d == 0.0:
        return
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    facing_out = math.atan2(uy, ux) + math.pi
    draw_triangle(surf, constants.RESCUE, pos, facing_out, constants.RESCUE_SIZE, cam)
    # Flames: build in world-space along the outward axis, then transform.
    for offset in (-5, 5):
        base_w = (
            pos[0] + ux * 10 + px * offset,
            pos[1] + uy * 10 + py * offset,
        )
        flame_len = 14 + random.uniform(-2, 4)
        tip_w = (base_w[0] + ux * flame_len, base_w[1] + uy * flame_len)
        pygame.draw.line(surf, constants.FLAME, cam.to_screen(base_w), cam.to_screen(tip_w), 2)


def draw_stranded(
    surf: pygame.Surface,
    pos: tuple[float, float],
    cam: Camera,
    facing: float = constants.STRANDED_FACING,
) -> None:
    draw_triangle(surf, constants.STRANDED, pos, facing, constants.STRANDED_SIZE, cam)


def draw_tug(
    surf: pygame.Surface,
    pos: tuple[float, float],
    facing: float,
    thrusting: bool,
    cargo: bool,
    cam: Camera,
) -> None:
    if cargo:
        draw_triangle(surf, constants.STRANDED, pos, facing, constants.STRANDED_SIZE, cam)
    tug_pos = tug_visual_center(pos, facing, cargo)
    draw_triangle(surf, constants.TUG, tug_pos, facing, constants.TUG_SIZE, cam)
    if thrusting:
        back = facing + math.pi
        base_w = (tug_pos[0] + math.cos(back) * 8, tug_pos[1] + math.sin(back) * 8)
        tip_w = (tug_pos[0] + math.cos(back) * 20, tug_pos[1] + math.sin(back) * 20)
        pygame.draw.line(surf, constants.FLAME, cam.to_screen(base_w), cam.to_screen(tip_w), 2)


def draw_capture_halo(
    surf: pygame.Surface, pos: tuple[float, float], radius: float, cam: Camera
) -> None:
    sx, sy = cam.to_screen(pos)
    pygame.draw.circle(surf, constants.HALO, (int(sx), int(sy)), int(cam.s(radius)), 1)


# Translucent filled wavy disks layered to form the solar flare field. Each
# disk has a sinusoidally modulated radius and slowly rotates via phase.
# Layered over each other on alpha surfaces, overlap regions composite
# darker/more saturated near the star, while orbit guides remain readable
# because there are no competing line strokes. Sized in fractions of
# STRANDED_ORBIT_RADIUS so the field always stays inside the orbit guide.
_FLARE_LAYERS: tuple[tuple[float, float, int, float, tuple[int, int, int]], ...] = (
    # (r_ratio, amp_ratio, lobes, phase_speed, base_rgb) — outer→inner
    (0.92, 0.040, 9, +0.45, (255, 210, 110)),
    (0.78, 0.055, 7, -0.55, (255, 170,  80)),
    (0.62, 0.070, 5, +0.65, (255, 130,  60)),
    (0.45, 0.080, 4, -0.80, (255,  95,  45)),
    (0.28, 0.100, 3, +1.00, (255,  60,  40)),
)


def draw_radiation_zone(
    surf: pygame.Surface, cam: Camera, t: float, intensity: float
) -> None:
    """Animated solar flare field: layered translucent wavy disks centered on
    the star. `t` is wall-clock seconds so waves move regardless of game
    pause; `intensity` in [0, 1] scales overall opacity for the ramp-in.
    """
    if intensity <= 0.0:
        return
    cx, cy = cam.to_screen(constants.STAR_POS)
    r_max_screen = cam.s(constants.DAMAGE_DANGER_RADIUS)
    if r_max_screen < 6:
        return
    diameter = int(2 * r_max_screen + 8)
    overlay = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    ocx = diameter // 2
    ocy = diameter // 2

    n_pts = 96
    for r_ratio, a_ratio, lobes, speed, base in _FLARE_LAYERS:
        r_base = r_max_screen * r_ratio
        amp = r_max_screen * a_ratio
        phase = t * speed
        # Each layer breathes: alpha pulses with a slow sine, scaled by intensity.
        alpha = int((36 + 14 * math.sin(t * 1.7 + lobes * 0.4)) * intensity)
        if alpha <= 0:
            continue
        color = (base[0], base[1], base[2], max(0, min(255, alpha)))

        points = []
        for i in range(n_pts):
            theta = 2.0 * math.pi * i / n_pts
            r = r_base + amp * math.sin(lobes * theta + phase)
            points.append((ocx + r * math.cos(theta), ocy + r * math.sin(theta)))

        # Reuse one overlay across all layers: clear, draw filled polygon, blit.
        # Layered blits alpha-composite onto the screen, so overlapping disks
        # darken naturally toward the star.
        overlay.fill((0, 0, 0, 0))
        pygame.draw.polygon(overlay, color, points)
        surf.blit(overlay, (cx - ocx, cy - ocy))


def wrap_segments(
    segments: list[tuple[str, tuple[int, int, int]]],
    max_chars: int,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Greedy word-wrap a list of (text, color) segments to lines of at most
    `max_chars`. Monospace assumed — width is char count. Segment colors are
    preserved across line breaks. Empty input returns `[[]]` (one blank line).
    Words longer than `max_chars` are not split; they overflow as one token.
    """
    if not segments:
        return [[]]
    if max_chars <= 0:
        return [list(segments)]

    tokens: list[tuple[str, tuple[int, int, int]]] = []
    for text, color in segments:
        i = 0
        while i < len(text):
            if text[i].isspace():
                tokens.append((text[i], color))
                i += 1
            else:
                j = i
                while j < len(text) and not text[j].isspace():
                    j += 1
                tokens.append((text[i:j], color))
                i = j

    lines: list[list[tuple[str, tuple[int, int, int]]]] = []
    current: list[tuple[str, tuple[int, int, int]]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        while current and current[-1][0].isspace():
            t, _ = current.pop()
            current_len -= len(t)
        lines.append(current)
        current = []
        current_len = 0

    for tok_text, tok_color in tokens:
        tok_len = len(tok_text)
        if tok_text.isspace():
            if not current:
                continue  # never start a line with whitespace
            if current_len + tok_len > max_chars:
                flush()
            else:
                current.append((tok_text, tok_color))
                current_len += tok_len
        else:
            if current_len + tok_len > max_chars and current:
                flush()
            current.append((tok_text, tok_color))
            current_len += tok_len

    if current:
        flush()

    return lines or [[]]


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
    title_color: tuple[int, int, int] = constants.HUD_ACTION,
) -> None:
    """Centered modal: title, segmented body lines, and a key-highlighted prompt."""
    screen_w, screen_h = surf.get_size()

    title_surf = title_font.render(title, True, title_color)
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
