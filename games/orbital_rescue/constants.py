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

MAX_TRAIL_POINTS = 600       # ~10 s at 60 Hz sim
MAX_PILOT_SECONDS = 180.0

ARRIVAL_START_RADIUS = 700.0         # rescue ship enters from off-screen right
ARRIVAL_DURATION = 3.0               # seconds to settle into station-keeping
PHASE_ANIM_DURATION = 5.0            # seconds — shared duration for the
                                     # game-driven docking and returning
                                     # animations (DOCKED, RETURNING)
STRANDED_FACING = -math.pi / 2       # disabled vessel — fixed pointing "up"

CAPTURE_RADII: dict[str, float] = {
    "small": 7.0,
    "medium": 15.0,
    "large": 30.0,
}
CAPTURE_CYCLE: tuple[str, ...] = ("small", "medium", "large")

TUG_THRUST = 288.0                     # px/s²  (old 0.08 px/frame² × 60²)
TUG_ROT_SPEED = math.radians(180.0)    # rad/s  (old 3°/frame × 60)
DOCK_RADIUS = 18.0
