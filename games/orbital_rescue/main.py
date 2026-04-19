"""Entry point — bootstraps the async game loop defined in `game`."""

import asyncio

import pygame  # needed here so pygbag detects the dep and installs pygame-ce

from game import run

_ = pygame  # silence unused-import warnings


async def main() -> None:
    await run()


asyncio.run(main())
