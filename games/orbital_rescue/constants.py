"""Shared constants for Orbital Rescue — layout, tuning, and palette.

All per-second quantities use SI-ish units: pixels for distance, seconds for
time. `DT_SIM` is the fixed physics timestep; rendering is decoupled.
"""

import math

WINDOW_SIZE = (1280, 720)
FPS = 60                     # render cap, informational on the browser target
DT_SIM = 1.0 / 60.0          # fixed physics timestep, seconds
MAX_FRAME_DT = 0.25          # clamp on real elapsed time per frame to avoid
                             # catch-up spirals after a pause / tab background

# Layout: a square viewport on the right holds the gameplay; the leftover
# rectangle on the left is the HUD/info panel. World math stays anchored on
# STAR_POS = (640, 360); rendering anchors that point at VIEWPORT_CENTER on
# screen so the star sits in the middle of the viewport.
VIEWPORT_SIZE = WINDOW_SIZE[1]                       # 720 — square
SIDE_PANEL_WIDTH = WINDOW_SIZE[0] - VIEWPORT_SIZE    # 560
VIEWPORT_X = SIDE_PANEL_WIDTH
VIEWPORT_Y = 0
VIEWPORT_RECT = (VIEWPORT_X, VIEWPORT_Y, VIEWPORT_SIZE, VIEWPORT_SIZE)
VIEWPORT_CENTER = (VIEWPORT_X + VIEWPORT_SIZE // 2, VIEWPORT_Y + VIEWPORT_SIZE // 2)

BG = (5, 5, 15)
STAR_CORE = (230, 140, 60)
STAR_CORONA = (180, 100, 40)
STAR_GLOW = (120, 60, 25)
STAR_OUTER = (40, 20, 10)
STRANDED = (255, 80, 200)            # neon magenta distress beacon
RESCUE = (60, 255, 120)              # neon green safe haven
TUG = (0, 240, 255)                  # electric cyan hardened ferry
FLAME = (255, 170, 80)
# Trail and capture halo share the tug's color so they read as part of it.
TRAIL = TUG
ORBIT_GUIDE = (35, 35, 55)
HALO = TUG
HUD = (180, 180, 200)
HUD_OK = (170, 240, 180)
HUD_BAD = (240, 150, 150)
HUD_ACTION = (240, 220, 140)
LOST_BOUNDARY = (240, 60, 60)        # bright red core of the lost-in-space ring
LOST_BOUNDARY_GLOW = (120, 30, 30)   # outer haze for the glow effect
LOST_BOUNDARY_HALO = (60, 15, 15)    # outermost faint ring
PANEL_BG = (10, 12, 25)              # slightly lighter than BG so the panel reads as a discrete UI surface
PANEL_DIVIDER = (40, 50, 80)         # vertical line between panel and viewport
PANEL_HEADER = (130, 135, 160)       # dim caps label color for section headers

STAR_POS = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
STAR_RADIUS = 32                     # hard core — crash boundary
STAR_CORONA_RADIUS = 46
STAR_GLOW_RADIUS = 62
STAR_OUTER_RADIUS = 82
STRANDED_ORBIT_RADIUS = 220
RESCUE_ORBIT_RADIUS = 320
RESCUE_POS = (STAR_POS[0] + RESCUE_ORBIT_RADIUS, STAR_POS[1])

# Ship sizes (triangle "radius" in pixels). Story framing: rescue and
# stranded are peer-sized capital ships; the tug is a small hardened ferry.
RESCUE_SIZE = 18
STRANDED_SIZE = 18
TUG_SIZE = 11

MAX_TRAIL_POINTS = 600       # ~10 s at 60 Hz sim while piloting

# Camera + lost-in-space boundary, both star-centered (radial).
# Inside INNER_RADIUS the view sits at zoom=1; past it the camera zooms out
# to keep the tug visible until tug distance hits LOST_RADIUS, at which
# point lost-in-space fires.
INNER_RADIUS = 360.0         # = window half-height — zoom engages as the
                             # tug crosses the top/bottom screen edge
LOST_RADIUS = 1500.0
MIN_ZOOM = INNER_RADIUS / LOST_RADIUS
ZOOM_SMOOTHING = 5.0         # exp rate; higher = camera reacts faster

ARRIVAL_START_RADIUS = 700.0         # rescue ship enters from off-screen right
ARRIVAL_DURATION = 3.0               # seconds to settle into station-keeping
PHASE_ANIM_DURATION = 5.0            # seconds — shared duration for the
                                     # game-driven docking and returning
                                     # animations (DOCKED, RETURNING)
STRANDED_FACING = -math.pi / 2       # disabled vessel — fixed pointing "up"
# Stranded ship starts opposite the rescue ship every mission so timing/score
# comparisons across attempts are meaningful.
STRANDED_START_ANGLE = math.pi

CAPTURE_RADII: dict[str, float] = {
    "small": 7.0,
    "medium": 15.0,
    "large": 30.0,
}
CAPTURE_CYCLE: tuple[str, ...] = ("small", "medium", "large")

# Maximum impact speed allowed at either dock — exceeding it is a hard fail.
# Cycled by the [v] settings key.
DOCK_SPEED_LIMITS: dict[str, float] = {
    "lenient": 200.0,
    "standard": 100.0,
    "strict": 50.0,
}
DOCK_SPEED_CYCLE: tuple[str, ...] = ("lenient", "standard", "strict")

# Combined difficulty: a single [d] key cycles the three values together.
# Level 1 = forgiving, level 3 = unforgiving. damage_mult scales both
# TUG_DAMAGE_RATE and STRANDED_DAMAGE_RATE proportionally — flares always
# fire after the delay, the rate just varies.
DIFFICULTY_PRESETS: dict[str, dict[str, str | float]] = {
    "1": {"capture": "large", "rv_max": "lenient", "damage_mult": 0.5},
    "2": {"capture": "medium", "rv_max": "standard", "damage_mult": 1.0},
    "3": {"capture": "small", "rv_max": "strict", "damage_mult": 2.0},
}
DIFFICULTY_CYCLE: tuple[str, ...] = ("1", "2", "3")

TUG_THRUST = 288.0                     # px/s²  (old 0.08 px/frame² × 60²)
TUG_ROT_SPEED = math.radians(180.0)    # rad/s  (old 3°/frame × 60)
DOCK_RADIUS = 18.0

# Fuel: a single tank covers the whole mission (no mid-mission refuel).
# Only thrust burns fuel; rotation is free. At 0 fuel the tug coasts.
FUEL_CAPACITY = 100.0
FUEL_BURN_RATE = 8.0                   # units per second of thrust

# Solar flare damage. The star is unstable; once it flares, solar energy
# saturates the inner zone. Flares fire INSTABILITY_DELAY seconds after
# launch and ramp in visually over INSTABILITY_FADE. DAMAGE_DANGER_RADIUS
# sits slightly outside STRANDED_ORBIT_RADIUS so the stranded vessel at
# r=220 is consistently inside the zone. The hardened tug takes damage
# at the same radii but at a much lower rate.
DAMAGE_CAPACITY = 100.0
TUG_DAMAGE_RATE = 0.5                  # %/s — hardened, takes damage slowly
STRANDED_DAMAGE_RATE = 1.5             # %/s — unshielded vessel
DAMAGE_DANGER_RADIUS = 230.0
INSTABILITY_DELAY = 10.0               # seconds after launch before flares fire
INSTABILITY_FADE = 1.5                 # seconds for the visual to ramp in
