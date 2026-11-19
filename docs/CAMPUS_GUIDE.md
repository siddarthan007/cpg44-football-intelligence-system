# Running CPG44 on your own ground

The models were trained on broadcast football (SoccerNet). A campus setup is
different: one fixed camera, lower and closer, often a phone with a wide lens,
different turf, kits and light, and a pitch that may not be 105 x 68 m. These
four steps close that gap. Each is a page in the app.

## 1. Camera check (`/prepare/camera`)

Drop a photo or a few seconds from the exact camera position. The page draws
what the system can see (painted lines in amber, players boxed by whether
they are large enough) and grades resolution, sharpness, exposure, pitch
coverage, visible lines and far-side player size.

Good campus setup: 4 to 6 m high on the halfway line (stand, balcony, mast),
tripod or clamp, 1080p at 25 to 60 fps, exposure locked on the grass, no zoom
during play. If far-side players are under ~30 px tall, add a second camera
for that half rather than zooming.

## 2. Pitch setup (`/prepare/pitch`)

1. **Frame**: drop a clip; scrub to a moment where the lines are clear.
2. **Pitch**: choose a preset (full, campus 100 x 64, 9 a side, 7 a side) or
   type the measured length and width; choose *Fixed on a tripod*.
3. **Landmarks**: tap a point on the small pitch, then click where it is on
   the frame. Corners, penalty-area and goal-area corners, halfway ends,
   penalty spots and the centre spot are offered (centre-circle extremes are
   not, because they are biased under perspective). Four points give a first
   fit; the overlay of the whole pitch appears immediately. Use 8 or more
   points spread over the frame: the lens curve is then estimated
   automatically and only kept if it predicts held-out points better
   (leave-one-out). The grade shows the median error of predicting each point
   from the others.
4. **Save**: *Snap to painted lines* refines the fit against the detected
   lines (accepted only if it improves). Save with a name; it appears in
   *Add footage*.

What the pipeline does with it: frames are undistorted with the lens curve,
the homography is held fixed (fixed camera), and it is re-snapped to the
lines every second only for small, clearly better corrections. Pitch size
flows into every metric, the board and the 3D view.

## 3. Adapt to your ground (`/workshop/adapt`)

1. Drop one or more clips from your ground; about 60 varied frames are taken
   from each (near-duplicates skipped).
2. *Let the model guess* pre-labels them (dashed boxes).
3. Review: drag to add, click to select, keys 1-4 set the class, B adds a
   ball with one click, Delete removes, Enter saves and moves on. Ten
   reviewed frames is the minimum; 60-150 is a good first round.
4. *Train* fine-tunes the current best model on native-scale tiles of your
   frames plus a replay of broadcast tiles (so it does not forget), holding
   out the last 20 % of your reviewed frames in time.
5. The page compares the current and adapted models on those held-out frames.
   *Use the adapted model* appears only if it is better.

## 3b. Futsal and small-sided pitches

Pitch setup has a Futsal preset (40 x 20 m). Futsal markings follow the FIFA
Futsal Laws: 3 m goals, a 3 m centre circle, and a penalty area made of two
6 m quarter circles; the landmarks you click change accordingly (the arc ends
on the goal line replace the box corners). xG uses the 3 m goal. The xG model
itself was fitted to 11-a-side shots, so futsal shot values are indicative.

## 3b2. Where to put cameras on a futsal court

Worked out for a 40 x 20 m court and a 1080p phone, with the height in pixels
of a player at the farthest point (the detector is reliable from about 30 px,
comfortable above 45 px):

| Placement | View needed | Far player, main lens (78°) | Far player, ultra-wide (110°) |
|---|---|---|---|
| Corner, 4 m up, 1 m outside | 86° | 46 px, but the lens cannot see the whole court | 26 px, whole court |
| Halfway line, 4 m up, 3 m back | 163° | not possible | not possible |
| Halfway line, 8 m up, 10 m back (balcony) | 127° | not possible | 33 px |
| **Two corners, diagonally opposite, 4 m up** | 86° each | every player within about 22 m of one camera | 55 px or more |

So: one camera works best in a corner, as high as possible, on the
ultra-wide lens with its distortion corrected in Pitch setup. Two cameras in
opposite corners are clearly better, because every player is close to one of
them.

### Two cameras

Calibrate each camera separately in Pitch setup (same court, each its own
saved calibration), record both, analyse each video on its own, then fuse:

```bash
PYTHONPATH=. python -m soccer_analytics.multicam \
    --analysis data/analyses/cam-a --calibration configs/calibrations/turf-a.yaml --video cam_a.mp4 \
    --analysis data/analyses/cam-b --calibration configs/calibrations/turf-b.yaml --video cam_b.mp4 \
    --out data/analyses/turf-fused
```

The two recordings are aligned from their sound (a whistle or ball strike
heard by both phones), players are matched on the court, and each position is
averaged with more weight on the nearer camera. In a simulated two-corner
futsal game the fused board had every player in view and 35 % lower position
error than one camera alone (`tests/test_multicam.py`).

## 3c. A phone as the pitch camera (`/workshop/cameras`)

Press *Show pairing code*, scan it with the phone and press Start. Fix the
phone sideways on a tripod as high as possible.

* The phone page is served by the relay over HTTPS, the only way a phone
  browser will open its camera. Only the connection set-up passes through the
  relay; the video goes straight from the phone to the laptop over WebRTC
  (hardware H.264, about 100 to 200 ms on the same Wi-Fi).
* If no direct path exists (some campus networks block it), the phone falls
  back to JPEG frames through the relay at up to 15 per second; the relay keeps
  only the newest frame so a slow link lowers the frame rate instead of adding
  delay. The status line says which path is in use.
* The QR code carries a per-room key that expires after 4 hours, never the
  relay token.
* Tested here with a simulated camera in Chromium: direct link at 1920 x 1080,
  relay fallback at 960 x 540, no frame queued behind another.

## 4. Add footage and review

*Add footage* with the saved calibration; the match opens in *Matches* with
the board (moment, options, shape, heat, passes, 3D), the match strip,
findings, the game plan and anything left to check.

## Accuracy you should expect, and how to check it

Measure on your ground before relying on numbers: walk a player along a
taped 30 m line and compare the reported distance; place cones at known
positions and compare the board. The protocol is in
`docs/CAPSTONE_ACCURACY_PROTOCOL.md`.
