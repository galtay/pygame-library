"""Pure geometric helpers shared by simulation and rendering.

No pygame dependency — keeps `game` from having to import `render` for layout
math.
"""

from __future__ import annotations

import math

CARGO_TUG_OFFSET = 12.0


def tug_visual_center(
    craft_pos: tuple[float, float], facing: float, cargo: bool
) -> tuple[float, float]:
    """Where the tug triangle is drawn — offset backward when towing cargo."""
    if not cargo:
        return (float(craft_pos[0]), float(craft_pos[1]))
    return (
        craft_pos[0] - math.cos(facing) * CARGO_TUG_OFFSET,
        craft_pos[1] - math.sin(facing) * CARGO_TUG_OFFSET,
    )
