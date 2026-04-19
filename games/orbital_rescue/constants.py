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

MAX_TRAIL_POINTS = 1200      # ~20 s at 60 Hz sim
MAX_PILOT_SECONDS = 180.0

DIFFICULTIES: dict[str, float] = {
    "easy": 30.0,
    "normal": 15.0,
    "hard": 7.0,
}

TUG_THRUST = 288.0                     # px/s²  (old 0.08 px/frame² × 60²)
TUG_ROT_SPEED = math.radians(180.0)    # rad/s  (old 3°/frame × 60)
DOCK_RADIUS = 18.0
