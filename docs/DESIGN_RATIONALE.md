# Interface design rationale (November 2026 overhaul)

## Who it is for and what they do

A college coach and analyst on matchday, then at a desk afterwards. The flow
follows the session, not the software modules:

1. **Session board**: can we start? (camera, calibration, kits, vests, relay),
   and what is still waiting from last time.
2. **Camera check**: will this camera position support tracking? One frame in,
   a verdict and concrete fixes out.
3. **Pitch & kits**: calibration and kit colours.
4. **Squad & vests**: pair each vest with one player.
5. **Live touchline**: positions, effort, pulse, and the few findings worth
   acting on now (`coach_insights`).
6. **Match review**: footage, board, match strip, findings, the review queue,
   players and the side-by-side.
7. **Evidence ledger**: the measured results behind the report objectives.

Setup tools that are rarely used (model training, camera sources, runtime,
vest flashing) live in *Workshop* and *Vest lab*, out of the matchday path.

## Second pass (12 Nov 2026, afternoon): less text, more depiction

Feedback asked for less clutter and nothing that reads as generated. Removed:
eyebrow labels, numbered section headers, coloured-stripe cards, "Try:"
prefixes, uppercase badges, dot and dash separators, the breadcrumb bar.
Replaced with: icon tabs on the board, URL-backed tabs for the lower panel
(findings, plan, to check, players, teams, shots) so state survives reload
and can be linked, confidence as three bars instead of a word, health as
three icons with tooltips, before/after bars on the Evidence page, a drawn
camera-placement diagram, and the camera check's own overlay of what it saw.
The board shades what the camera sees and fades players who just left the
frame, which answers "why is a player missing" without a sentence.

## Visual language: a touchline notebook

The interface borrows from the tools coaches already use: a paper notebook
and a tactics board. Paper surfaces (`--paper`), ink rules instead of heavy
cards, and a sage pitch drawn like a whiteboard with ink lines. Typography is
Archivo for words and IBM Plex Mono for every number and timecode, so values
line up and read as measurements.

Colour carries meaning, never decoration: cobalt and vermilion for the two
sides, amber for the ball, and a magenta *pulse* accent reserved for anything
that comes from the vest, so physiological data is never mistaken for camera
evidence. Warning amber marks things that need a human decision.

## What is original here, and how it differs from Replayfield

Replayfield (replayfield.com) was studied as a reference for the *problems*
an amateur-football video product must solve: camera guidance, uncertain
events, replays and shareable cards. CPG44 does not reuse its layout, copy,
labels, colours or visuals:

| Concern | Replayfield (public page) | CPG44 |
|---|---|---|
| Overall look | Dark match room, neon pitch | Light paper notebook, ink-on-sage tactics board |
| Uncertain events | Goal / No-goal buttons on a goal list | *Your call* queue covering identity joins, side assignment and shots, each worded for that decision ("Same player / Different") |
| Timeline | Possession and goal markers | **Match strip** with lanes for ball control, moments, running effort per side, **vest pulse** and decisions |
| 3D | Goal replay | Any moment; ball height from a physics fit with on-screen honesty note and camera presets |
| Analysis output | Stats and cards | **Findings with evidence**: each card states the observation, numbers, confidence, the moments to watch and the report objective it serves |
| Wearables | None | Vest heart rate and movement fused with camera speed (drift, recovery, link integrity) |
| Trust | Accuracy claims in FAQ | Evidence chips on every analysis and an Evidence ledger page backed by benchmark files |

## Rules the interface enforces

- Missing stays missing: no calibration means no metres, no clean pulse means
  no heart rate; the empty states explain what to do next.
- Inferred values are visibly inferred (filled ball gaps, interpolated
  positions, 3D figures are "drawn models").
- A finding never appears without its evidence and confidence.
- Keyboard: the match strip is a slider (arrow keys, shift for 10 s); every
  control is a real button or link; focus is a visible amber ring.
- Works down to phone width; motion is reduced when the OS asks.
