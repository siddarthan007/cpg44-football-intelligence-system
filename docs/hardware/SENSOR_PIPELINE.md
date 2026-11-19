# Vest data path: what was measured, what was wrong, what changed

All numbers below come from the 21 recorded vest sessions in
`data/live_sessions/` (August 2026, prototype vest, relay at
cpg44.nivaspms.com) unless a test or synthetic benchmark is named.

## 1. Delivery: half of the samples never arrived

| | Designed | Delivered (37 min session) |
|---|---|---|
| IMU | 50 Hz | 23.6 Hz |
| Pulse (PPG) | 25 Hz | 11.5 Hz |
| Vest counter `dropped_samples` | 0 | up to 106,502 per boot |

The firmware sent one HTTPS POST per batch and waited for the reply. Through
Cloudflare a POST took 0.5 to 1 s and carried at most 32 samples, so the
sender topped out near 35 samples/s while the sensors produced 75/s; the rest
expired in the queue. The relay accepts 100 observations per POST, so the
batch is now 96 (about 34 KB of JSON, 12 KB of RAM), which keeps up even at
1.2 s per round trip (`firmware/wearable_stream/wearable_stream.ino`,
`RELAY_BATCH_CAPACITY`).

Consequence found while replaying the logs through the hub: 22 % of pulse
intervals were gaps, so almost every 4 s pulse window failed the continuity
check and **the recorded sessions produced no heart rate**. The hub now
bridges gaps under 0.32 s on a uniform grid (and rejects windows more than
15 % bridged), but the real fix is the delivery rate above. Heart-rate
accuracy still has to be measured in the field against a chest strap.

## 2. Clocks: the laptop was up to 0.77 s behind

The vest stamps samples with SNTP time. Comparing relay receive time with the
laptop's receive time gave negative medians in every session (-27 to
-767 ms): impossible unless the laptop clock is behind. WSL2 clocks drift after
sleep. Video frames are stamped with the laptop clock, so vest and video were
misaligned by that amount.

* The relay now serves `GET /api/v1/time`; the hub probes it six times every
  30 s and keeps the shortest round trips (Cristian 1989; NTP clock filter).
  Vest timestamps are mapped onto the laptop clock with the measured offset;
  the uncertainty is half the best round trip
  (`sensors/hub.py: RelayClockOffset`, test
  `test_relay_clock_offset_uses_shortest_round_trips`).
* For recorded footage the remaining offset is measured from the data itself
  by the vest/track correlation (section 5).

## 3. IMU

| Problem | Fix |
|---|---|
| Gyro-bias auto-calibration kept collecting after the first still sample (and after 30 samples regardless), so real rotation leaked into the bias | `GyroBias`: only 1 s windows with gyro standard deviation < 0.01 rad/s and gravity-only acceleration; slow tracking afterwards for thermal drift. Replay of a 37 min session: 138 still windows, bias -2.1 / 0.2 / 0.1 deg/s |
| Accelerometer offset and scale never estimated in the field | `AccelAutoCal` (Frosio et al. 2009): fits offset and scale per axis from still poses in different orientations; accepted only when the fit residual is under 2 % g. Test recovers a 0.25 m/s^2 offset and 3 % scale error |
| "Speed" from acceleration magnitude x 1.4 | Removed. An accelerometer cannot measure speed; speed comes from GNSS Doppler or the camera |
| PlayerLoad from a thresholded linear acceleration | Standard definition (Barrett et al. 2014): accumulated change of acceleration in g / 100, plus per-minute rate |
| Status packet claimed 100 Hz, +-8 g, +-1000 deg/s | Reports the configured 50 Hz, +-16 g, +-2000 deg/s, DLPF 21 Hz |
| Fallback sample interval 10 ms | 20 ms (50 Hz) |

## 4. Pulse, SpO2, GNSS, battery

* **SpO2.** The ratio of ratios R maps to SpO2 through a curve specific to
  the LEDs, photodiode and site. The Maxim reference curve is a starting
  point only. `POST /api/calibrate/spo2 {"reference_pct": 98}` on the hub
  pairs the current still reading with a finger pulse oximeter: one pair
  gives an offset, three or more pairs spanning 3 % give a line. SpO2 is still
  only shown in still windows and is labelled uncalibrated until paired.
* **GNSS.** The NEO-6M is set to 5 Hz with only RMC and GGA (7.3 kbit/s,
  fits 9600 baud), and each new fix is sent as it arrives instead of on a 1 s
  timer. The hub runs a constant-velocity Kalman filter in local metres using
  the receiver's Doppler speed and course (Townshend et al. 2008), gates
  multipath jumps with a chi-square test, ignores fixes with fewer than five
  satellites or HDOP above 4, and integrates distance from filtered speed above
  walking pace, so standing still no longer adds jitter as distance. Synthetic
  test: 6.0 m/s recovered within 0.4 m/s, distance within 8 m over 119 m.
* **Battery.** The design's MAX17048 fuel gauge (I2C 0x36) is now read every
  2 s: cell voltage, state of charge and rate. The hub derives time left from
  the last 10 minutes of charge and warns below 15 % or 20 minutes.

## 5. Which player wears the vest

`sensors/association.py` correlates the vest's movement intensity
(gravity-free acceleration, RMS over 1 s) with each tracked player's speed,
over lags of +-2 s (+-10 s when the recording start comes from file metadata).
The best track is the wearer; the lag is the residual clock offset. Several
vests are assigned jointly so two vests never claim one player, short track
fragments are not candidates, and a match is marked confident only with a
score of at least 0.35 and a margin of 0.08 over the next player. A GNSS fix,
when present, adds a position term.

Synthetic test (20 players, one vest, 0.8 s clock offset, a third of the
players out of view for 50 s): wearer found with correlation 0.97 against 0.29
for the next player, offset recovered as 0.8 s. Real paired vest-and-video
recordings are needed to report field accuracy.

The squad page links vest numbers and shirt numbers to names. The review
cross-checks the two cues: if the vest's player shows a different shirt number
than the squad lists, the pairing goes to To check.

## 5b. Chest ECG (AD8232)

Heart rate now comes from a single-lead chest ECG when the vest carries one.
The firmware samples it at 250 Hz from its own task and sends 50-sample
blocks; the hub finds beats with Pan-Tompkins (band-pass 5 to 15 Hz,
derivative, squaring, 150 ms integration, adaptive threshold, 250 ms
refractory period), drops implausible beat intervals (outside 30 to 220 bpm
or more than 30 % from their neighbours) and reports RMSSD in still windows of
30 s or more. Electrode-off (the AD8232's LO pins) and flat signals are
reported as such, never as a number.

Synthetic recordings with known beats (`tests/test_ecg.py`): every beat found
within 2 ms; heart rate within 2.5 bpm at 60, 150 and 185 bpm including step
impacts and muscle noise. The end-to-end check (`scripts/e2e_live.py`), with a
simulated vest posting through a local relay, read 150.0 bpm for a 150 bpm
signal. Field accuracy needs a session beside a chest-strap reference.

## 6. Build check

The firmware compiles for the ESP32-S3 with arduino-esp32 3.3.12 and no
warnings from the sketch: 1,062,669 bytes of flash (81 %) and 61,952 bytes of
static RAM (18 %), leaving 266 KB for the larger relay batches. Behaviour on
the vest (delivery rate, 5 Hz fixes, fuel-gauge readings) still has to be
confirmed on hardware; the status packet now reports `gps_rate_hz`,
`fuel_gauge` and `dropped_samples` so that check needs no extra tools.

## 7. Run it

```bash
# review footage with a vest log; start time from the file or given explicitly
PYTHONPATH=. python -m soccer_analytics.match_analysis --video match.mp4 \
    --weights runs/detect/soccernet/weights/best.pt --calibration configs/calibrations/ground.yaml \
    --out data/analyses/match --vest data/live_sessions/<session>/wearable_raw.ndjson
```

## References

Barrett, Midgley & Lovell (2014) PlayerLoad: reliability, convergent validity
and influence of unit position. *Int J Sports Physiol Perform* 9(6).
Cristian (1989) Probabilistic clock synchronization. *Distributed Computing* 3.
Frosio, Pedersini & Borghese (2009) Autocalibration of MEMS accelerometers.
*IEEE Trans Instrum Meas* 58(6).
Townshend, Worringham & Stewart (2008) Assessment of speed and position
during human locomotion using nondifferential GPS. *Med Sci Sports Exerc* 40(1).
Teixeira, Jung & Savvides (2010) Tasking networked CCTV cameras and mobile
phones to identify and localize multiple people (PEM-ID). *UbiComp*.
Cabrera-Quiros, Tax & Hung (2018) Gestures in-the-wild: detecting
conversational hand gestures in crowded scenes using a multimodal fusion of
bags of video trajectories and body worn acceleration. *IEEE Trans Multimedia*.
Maxim Integrated, MAX30102 reference design, ratio-of-ratios SpO2 curve.
