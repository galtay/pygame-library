# AGENTS.md

## Project

Python game built with [pygame-ce](https://github.com/pygame-community/pygame-ce) (pygame Community Edition).

## Environment

- Python environment is managed with [uv](https://docs.astral.sh/uv/).
- Add dependencies with `uv add <pkg>`; run commands with `uv run <cmd>`.
- The game dependency is `pygame-ce` (not `pygame`). Do not install both.

## Design

Follow SOLID principles:

- **S**ingle Responsibility — each class/module does one thing.
- **O**pen/Closed — extend via new types, not by editing existing ones.
- **L**iskov Substitution — subtypes must be drop-in replacements for their base.
- **I**nterface Segregation — prefer small, focused protocols/ABCs over fat ones.
- **D**ependency Inversion — depend on abstractions; inject collaborators rather than constructing them inside consumers.

## Conventions

- Keep game loop, rendering, input, and entity logic in separate modules.
- Prefer composition over inheritance for entity behavior.
- Type hints on public functions and class attributes.

## Packaging for the web (pygbag → itch.io)

Games in this repo are shipped as WebAssembly via
[pygbag](https://github.com/pygame-web/pygbag). Several non-obvious pitfalls
apply to *every* game we port:

1. **At module-load time, `pygame` is a stub.** Any reference to `pygame.K_1`,
   `pygame.Surface`, `pygame.font.Font`, etc. evaluated during `import` raises
   `AttributeError`. Move them inside functions that run after
   `pygame.init()`, or add `from __future__ import annotations` so type hints
   stay as strings.

2. **pygbag only scans `main.py` for imports.** If main.py delegates to
   another module that does the `import pygame`, pygbag never installs
   `pygame-ce`. Always `import pygame` at the top of `main.py` (even if unused
   there), or add a PEP 723 `# /// script` header.

3. **Do not pass `--git` to pygbag.** It forces the bleeding-edge CDN path
   (`pygame-web.github.io/pygbag/0.0/`) which 404s on `cpython312/main.js`.
   Use the stable default: `uv run pygbag --build <game-folder>`.

4. **`pygame.font.SysFont(...)` is unreliable in the browser.** Prefer
   `pygame.font.Font(None, size)` — the bundled default font, guaranteed
   available.

5. **Use `WINDOW_SIZE = (1280, 720)`.** That matches pygbag's default
   framebuffer, so the browser scales the canvas at integer ratios — no
   fuzzy text, no reshape on browser resize.

6. **Game loop must be `async`** and call `await asyncio.sleep(0)` after
   each `pygame.display.flip()`. `main.py` runs the coroutine with
   `asyncio.run(main())`.

7. **Decouple simulation from frame rate.** Use a fixed-timestep accumulator
   (`DT_SIM = 1/60 s`) driven by real elapsed time; render once per loop.
   Browser refresh rates vary (60/120/144 Hz, tab throttling), and this keeps
   gameplay identical across hardware.

### Build + upload

```bash
uv run pygbag --archive --build games/<name>
```

produces `games/<name>/build/web.zip`. On itch.io: upload the zip, check
"This file will be played in the browser", set viewport to
`1280 × (720 + ~60)` to leave room for the itch footer, turn mobile-friendly
off for keyboard games.
