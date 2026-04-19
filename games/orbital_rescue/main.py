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

import asyncio

import pygame  # needed here so pygbag detects the dep and installs pygame-ce

from game import run

_ = pygame  # silence unused-import warnings


async def main() -> None:
    await run()


asyncio.run(main())
