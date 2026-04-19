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
