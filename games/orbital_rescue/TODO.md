# TODO — Orbital Rescue

## Gameplay

- **Stellar radiation damage mechanic.** When stellar damage is ON:
  stranded ship accumulates radiation dose continuously (it's in a low,
  hot orbit); the hardened tug accumulates dose only when its path dips
  below a danger radius. Failure if either vessel's dose maxes out. HUD
  needs a dose meter. Ties into the mission-briefing framing.

- **Relative-velocity docking failure threshold.** Live relative speed
  (`relative-v stranded` / `relative-v rescue`) already tracked in the HUD
  and locked at each capture. Still pending: above a threshold, the
  dock fails catastrophically (explosion / FAILED state) instead of
  succeeding. Natural second difficulty lever alongside capture radius.

## Audio

- **Basic 8-bit music.** Chiptune background track for ambience,
  ideally with phase-aware variation (arrival / outbound / docking /
  homebound / win / fail). Sound effects too: thrust, capture lock,
  dock chime, crash, rescue-complete fanfare.
