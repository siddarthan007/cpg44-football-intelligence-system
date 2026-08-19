# Wearable hardware — build & integration guide (CPG44)

Player-worn unit that streams IMU + HR/SpO2 + GPS to the analytics endpoint. The
software runs **vision-only today**; the wearable augments and cross-checks the
vision metrics when built.

**Flash this sketch (the one that matches the working board):**
[`firmware/wearable_stream/wearable_stream.ino`](firmware/wearable_stream/wearable_stream.ino)

It opens **TCP port 9000** and waits for the WSL sensor hub to connect. Do not
point it at `/ingest` — that is the older HTTP firmware in
`firmware/soccer_wearable/` (kept for reference).

```bash
python -m soccer_analytics.hub --esp32 <printed-ESP32-IP>
# dashboard: http://127.0.0.1:8081/
```

## Bill of materials
| part | role | bus |
|------|------|-----|
| ESP32-S3 WROOM N16R8 | MCU + Wi-Fi | — |
| GY-87 (MPU6050 + HMC5883L + BMP180) | accel/gyro (load), mag (heading), baro (altitude) | I²C |
| MAX30102 | heart rate + SpO2 | I²C |
| NEO-6M GPS | outdoor position/speed | UART |
| TP4056 (adjustable) | LiPo charging | — |
| NOVA 3000 mAh 3.7 V LiPo | power | — |
| MT3608 boost (add) | 3.7 V → 5 V for the ESP32 | — |

> Add one small **MT3608 boost converter**: a 3.7 V LiPo is below the dropout of
> the ESP32 dev-board regulator, so boost to 5 V and feed the 5V/VIN pin.

## Wiring

**I²C bus (shared)** — ESP32 `SDA=GPIO8`, `SCL=GPIO9` (adjust to your board silk):
```
GY-87  VCC→3V3   GND→GND   SDA→GPIO8   SCL→GPIO9
MAX30102 VIN→3V3 GND→GND   SDA→GPIO8   SCL→GPIO9
```
Both sensors sit on the same two I²C lines (different addresses: MPU 0x68, BMP180
0x77, HMC5883L 0x1E behind the MPU, MAX30102 0x57).

**GPS (UART)** — `GPS TX→GPIO18`, `GPS RX→GPIO17`, `VCC→3V3`, `GND→GND`, 9600 baud.

**Power chain:**
```
LiPo + ──▶ TP4056 B+      TP4056 OUT+ ──▶ MT3608 IN+ ──▶ MT3608 OUT+ (set 5.1 V) ──▶ ESP32 5V
LiPo − ──▶ TP4056 B−      TP4056 OUT− ──────────────────────────────────────────▶ ESP32 GND (common)
```
Sensors are powered from the ESP32 **3V3** pin. Charge over the TP4056 USB-C.
Prefer a TP4056 module **with DW01 battery protection** (over-discharge cutoff).

**Robustness (do these — they prevent the classic field failures):**
- **470–1000 µF electrolytic cap across the ESP32 5V/GND input.** Wi-Fi TX draws
  ~500 mA spikes; without the cap the boost converter sags and the ESP32
  brown-out-resets mid-match. This is the #1 ESP32 wearable failure mode.
- Single **common ground** star point; keep I²C wires <15 cm (GY-87 and MAX30102
  have onboard 4.7 kΩ pull-ups — do not add more).
- Strain-relieve every wire at the enclosure (hot glue / zip tie); vibration on a
  player breaks solder joints, not the code.
- GPS patch antenna facing the sky, away from the ESP32 antenna (~3 cm apart).
- Set MT3608 to 5.1 V under load BEFORE connecting the ESP32 (trim pot).

```
        ┌──────────┐   3V3   ┌──────── GY-87 (IMU/mag/baro)
 LiPo ─▶ TP4056 ─▶ MT3608 ─▶ ESP32-S3 ┼──────── MAX30102 (HR/SpO2)
        (charge)   (→5V)     │  UART   └──────── NEO-6M GPS
                             └── Wi-Fi ▶ endpoint
```

## Network (university Wi-Fi is NATed / AP-isolated)

The ESP32 makes **outbound** connections, so pick whichever fits the venue:

1. **Phone hotspot (recommended, outdoors).** ESP32 and laptop both join the
   hotspot. Set `ENDPOINT="http://<laptop-LAN-IP>:8000/ingest"`. Run the endpoint
   in-process on the laptop (below). Low latency, no university-NAT problems, and
   the hotspot's cellular data gives the ESP32 NTP time.
2. **Cloud relay** (works anywhere with internet). Run `run_relay()` on a small
   public VPS; ESP32 POSTs to `http://<vps>:8000/ingest`; the laptop subscribes to
   `ws://<vps>:8000/subscribe`. Both sides dial out, so NAT/isolation is irrelevant.
3. **ESP32 SoftAP** (isolated field, no internet). ESP32 becomes the AP, laptop
   joins it. No NTP → the server stamps arrival time (already handled).

Run the laptop endpoint + pipeline together:
```bash
python -m soccer_analytics.realtime --video 0 --weights runs/detect/soccernet/weights/best.pt \
    --calibration campus.yaml --wearable-endpoint 8000 --roster "7:7,10:10,4:4"
```
(`--video 0` = webcam; `--roster track:player,…` binds vision track ids to wearable
player ids, or `--roster-numbers jersey:player` for OCR auto-binding.)

## Wire format & sync (firmware ↔ server)

The firmware accumulates samples and POSTs them **in batches** (~5 samples per
500 ms): one HTTP request instead of five cuts radio on-time ≈5× (battery), and
a failed POST keeps the batch for the next flush — brief Wi-Fi dropouts lose
nothing (bounded at 25 samples, oldest dropped). Payload (fields optional except
`player_id`):
```json
[{"player_id":7,"t":1699999999.12,"hr":150,"spo2":97,
  "accel":[0.10,-0.20,0.99],"gyro":[1.2,-0.4,0.1],
  "altitude":312.5,"gps":[30.1,76.2],"source":"esp32"}]
```
Units: accel in g (±2 g), gyro °/s (±250), altitude m. **`t` is sent only when
NTP has synced** — an unsynced (millis-based) clock would never match the
video's wall-clock sync window and the data would be silently ignored; with `t`
absent the server stamps arrival time (~100 ms accuracy, fine for HR/SpO2).

## Sensor placement — accuracy vs comfort

- **Main unit + IMU:** upper back, between the shoulder blades, in a snug
  compression vest (the Catapult standard). Captures trunk PlayerLoad and
  accel/decel cleanly, stays out of the way.
- **MAX30102 (HR/SpO2):** needs firm, low-motion skin contact — mount on a
  **forearm/upper-arm strap** (short wire to the main unit) or an earlobe clip,
  **not** the back. Honest limitation: optical HR/SpO2 is noisy under running
  motion → treat it as **warm-up / rest / recovery** data. For accurate in-play
  HR, a chest-strap ECG (e.g. Polar H10) is the better optional upgrade.
- **NEO-6M GPS:** antenna facing up/out; ~2.5 m accuracy at 1 Hz. On-pitch, the
  **vision** positions are more accurate — GPS is for outdoor total-distance
  cross-checks and when a player is off-camera.

## How the two streams are used (fusion)

| metric | primary source | cross-check |
|--------|----------------|-------------|
| position, speed, distance | **vision** (Kalman) | GPS |
| metabolic power, HSR, sprints | **vision** (di Prampero) | — |
| accel/decel load, PlayerLoad | vision proxy | **IMU** (true Catapult PlayerLoad) |
| heart rate, SpO2 (internal load) | **MAX30102** | — |
| altitude / heading | BMP180 / HMC5883L | — |

The pipeline runs fully on vision alone; when a wearable sample is time-aligned to
a frame, `LoadEngine` adds true IMU PlayerLoad + HR/SpO2, and the injury model /
recommendation engine use both. Vision and wearable estimates of the same quantity
(e.g. distance: vision vs GPS) can be compared per player for validation.

## Outdoor & runtime

- 3000 mAh at ~120 mA average ≈ many hours; Wi-Fi TX is the main draw.
- Weatherproof the unit in a small vented pouch; keep the GPS patch antenna clear.
- Calibrate the pitch once per fixed campus camera (`soccer_analytics.calibrate`);
  a fixed tripod camera gives a stable homography → accurate metres.
