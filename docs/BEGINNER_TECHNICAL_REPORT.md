# How CPG44 works

A camera shows player, team and ball movement. The wearable measures motion,
a chest ECG and GPS. The project PC joins both using player ID and
time.

```text
wearable -> cpg44.nivaspms.com -> local sensor checks --+
camera or footage -> detector -> ByteTrack -> pitch map --+-> dashboard
```

The relay is required for wearable data. It keeps only a small memory window so
the PC can reconnect. It does not store match files.

## Wearable values

The firmware sends raw AD8232 ECG samples, MPU6050 motion and NEO-6M GPS. The
PC finds each heartbeat and calculates heart rate and heart-rate variability.
An electrode off the skin or a noisy signal gives a blank result instead of a
made-up value.

## Video values

YOLO detects football objects and ByteTrack joins detections over frames. Team
colour, jersey votes and manual review link a track to a player. Pitch
calibration is required for metres, speed, distance and team-shape measurements.

## Time and identity

The ESP32 uses network time and its monotonic timer to timestamp every sample.
The relay preserves that time and adds its own order number. The PC links a
wearable value only to the configured player and nearby video time.

## Coach use

The useful outputs include player distance, high-speed work, team width and
depth, possession evidence, repeat-sprint change, recovery heart-rate response
and an explained substitution watch. These are review aids, not medical
diagnoses.

Start with the steps in `docs/DEMO_TUTORIAL.md`.
