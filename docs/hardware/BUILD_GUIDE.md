# Building the CPG44 vest

This guide is for the parts on hand in November 2026:

ESP32-S3 DevKitC (ESP32-S3-WROOM-1 **N16R8**), GY-87 10-DoF (MPU6050,
HMC5883L, BMP180), NEO-6M GPS with EEPROM, AD8232 ECG front end, TP4056
charger, TPS63020 buck-boost (3.3 V), Nova 105050 3.7 V 3000 mAh LiPo, SPDT
slide switch, 470 µF 25 V electrolytic capacitor, 100 kΩ 1 % resistors.

The firmware in `firmware/wearable_stream` already matches the wiring below.

## 1. What each part is for

| Part | Job in the vest | Verdict |
|---|---|---|
| ESP32-S3 N16R8 | Reads the sensors, timestamps every sample, sends through the relay | Keep. Build with the 16 MB flash and OPI PSRAM settings (the in-app flasher does this) |
| AD8232 | Single-lead chest ECG for heart rate and heart-rate variability | Keep. It replaces the optical pulse for heart rate: an ECG keeps beat timing while running, where optical sensors fail |
| GY-87 | MPU6050 accelerometer and gyroscope for PlayerLoad, impacts and orientation; magnetometer for heading | Keep. The BMP180 on the same board is not used (section 7) |
| NEO-6M | Position and Doppler speed when the camera cannot see the player | Keep. Set to 5 fixes per second by the firmware |
| TP4056 | Charges the LiPo from USB | Keep, **only the version with protection** (DW01A and FS8205A chips, pads marked B+, B-, OUT+, OUT-) |
| TPS63020 | Turns 3.0 to 4.2 V from the cell into a steady 3.3 V | Keep. It is the reason the vest runs to the bottom of the battery |
| 3000 mAh LiPo | Power | Keep. It must have its own protection board (most 105050 cells do) |
| SPDT slide switch | Power on and off | Keep. Use two of its three pins |
| 470 µF 25 V | Holds the 3.3 V rail up during Wi-Fi bursts | Keep. The 25 V rating is more than needed; a 470 µF 6.3 V low-ESR part is smaller if you buy again |
| 100 kΩ 1 % (two) | Battery voltage divider | Keep |

### Still needed

| Item | Why |
|---|---|
| 3-lead ECG cable with snap ends (3.5 mm plug for the AD8232 board) | Connects the electrodes |
| Gel snap electrodes for testing, then a chest strap with textile electrodes | Gel for the first trials; textile for repeated matches |
| 100 nF ceramic capacitors (5) and 10 µF ceramic capacitors (2) | Local decoupling at each module and the ADC filter |
| A 10 Ω resistor and a 10 µF capacitor (or a ferrite bead) | Clean supply for the AD8232 |
| JST-PH 2-pin connectors, silicone wire, heat shrink | Battery and harness connectors that survive tackles |
| Perfboard or a small carrier PCB, a 3D-printed or ABS pod (about 90 x 55 x 20 mm) | Mounting |

### Not needed

- The MAX30102 pulse oximeter from the first prototype. The chest ECG gives
  heart rate and heart-rate variability directly, keeps beat timing while
  running (the optical pulse failed in every recorded session) and the
  firmware no longer reads it.
- Any 5 V boost module or the DevKit's 5 V pin: everything runs at 3.3 V.
- Level shifters: every module here is 3.3 V compatible.
- The MAX17048 fuel gauge from the earlier design: the divider on GPIO 1 measures
  the battery instead (about 5 % accuracy, enough for "minutes left").

## 2. Power

```
LiPo + ─── TP4056 B+          TP4056 OUT+ ─── switch ─── TPS63020 VIN ──┐
LiPo - ─── TP4056 B-          TP4056 OUT- ───────────── TPS63020 GND   │
                                                                         │ 100k
                                    TPS63020 VOUT (3.3 V) ── 3V3 rail    ├──── GPIO 1 ── 100 nF ── GND
                                    TPS63020 GND ─────────── GND         │ 100k
                                                                         └──── GND
```

1. **Feed 3.3 V into the DevKit's 3V3 pin, not 5V or VIN.** The DevKit's own
   regulator needs about 4.3 V in, so a LiPo below 4 V would brown out the
   board through that path. Going straight to 3V3 bypasses it.
2. **Put the battery divider after the switch** (on TPS63020 VIN). Before the
   switch it would drain about 20 µA from the cell forever, even when off.
3. **Capacitors on the 3.3 V rail, close to the ESP32 3V3 and GND pins:**
   470 µF electrolytic (stripe to GND), plus 10 µF and 100 nF ceramics. Wi-Fi
   transmit bursts pull about 350 mA for a few milliseconds; without the bulk
   capacitor the ESP32 can reset on brown-out. Keep the ceramics: the
   converter's control loop is designed around them, and the electrolytic is
   an addition, not a replacement.
4. **100 nF on each module's VCC pin** (GY-87, NEO-6M, AD8232).

### Power budget

| Load at 3.3 V | Typical current |
|---|---|
| ESP32-S3, Wi-Fi on, sending every 0.5 s | 120 to 160 mA |
| NEO-6M tracking | 40 to 45 mA |
| MPU6050, magnetometer, BMP180 | 4 mA |
| AD8232 | 0.2 mA |
| LEDs on the modules | 5 to 10 mA |
| **Total** | **about 210 mA** |

From the cell: 210 mA x 3.3 V / (3.7 V x 0.92 converter efficiency) is about
205 mA, so 3000 mAh gives roughly **12 hours**, comfortably more than a match
plus warm-up. Removing the power LEDs saves about 5 %.

### Charging

The TP4056 charges at 1 A by default (its 1.2 kΩ program resistor), so the
3000 mAh cell fills in about 3.5 hours. **Switch the vest off while charging.**
With the load switched on, the charger sees the vest's current as charge
current and may never reach its stop condition. **Never charge while the vest
is worn:** the USB lead connects the electrodes, through the circuit, to a
mains-powered adapter. Charge outside the vest pocket, because the TP4056 gets
hot at 1 A.

## 3. Pin map (ESP32-S3 N16R8)

| ESP32 pin | Goes to | Notes |
|---|---|---|
| 3V3 | TPS63020 VOUT, all module VCC | see section 2 |
| GND | common ground | one star point near the ESP32 |
| GPIO 4 | GY-87 SDA | I2C |
| GPIO 5 | GY-87 SCL | I2C |
| GPIO 16 (RX) | NEO-6M TX | 9600 baud |
| GPIO 17 (TX) | NEO-6M RX | needed so the firmware can set 5 Hz |
| GPIO 2 | AD8232 OUTPUT | ADC1 channel 1 |
| GPIO 11 | AD8232 LO+ | electrode-off detect |
| GPIO 12 | AD8232 LO- | electrode-off detect |
| GPIO 1 | battery divider midpoint | ADC1 channel 0 |
| AD8232 SDN | 3V3 | always on (pull it low to put the AD8232 to sleep) |

Pins **not** to use on this board: GPIO 26 to 37 (internal flash and the octal
PSRAM of the R8 variant), 0, 3, 45 and 46 (boot strapping), 19 and 20 (USB),
43 and 44 (USB serial). Analog inputs must be on ADC1 (GPIO 1 to 10): ADC2
cannot be read while Wi-Fi is on.

## 4. GY-87

| Chip | I2C address | Used for |
|---|---|---|
| MPU6050 | 0x68 | acceleration ±16 g, rotation ±2000 °/s, 50 Hz, 21 Hz filter |
| HMC5883L or QMC5883L | 0x1E or 0x0D | magnetic heading |
| BMP180 | 0x77 | not used |

- Power the board from **VCC_IN at 3.3 V**. Its own regulator then gives the
  chips about 3.1 V, which all three accept. The board has I2C pull-ups.
- The HMC5883L sits behind the MPU6050. The firmware switches the MPU6050 to
  "bypass" so the magnetometer appears on the main bus.
- Many GY-87 boards now carry a **QMC5883L** (address 0x0D, different
  registers) instead of the HMC5883L. The firmware detects either one and
  prints which it found on the serial monitor.

## 5. AD8232 ECG

### Wiring

| AD8232 board | Connect to |
|---|---|
| 3.3V | 3V3 through 10 Ω, with 10 µF and 100 nF to GND at the board (a small filter against the converter's switching noise) |
| GND | GND |
| OUTPUT | GPIO 2 |
| LO+ / LO- | GPIO 11 / GPIO 12 |
| SDN | 3V3 |

Keep the OUTPUT wire short and away from the TPS63020 and the ESP32 antenna.
Twist it with a ground wire if it runs more than 10 cm.

### Electrodes for running

Use a chest lead, not the arm positions printed on the cable (RA, LA, RL refer
to arms). For a player who is running:

- **RA (white)**: right side of the chest, just below the collarbone.
- **LA (black)**: left side, on the ribs below the left pectoral muscle.
- **RL (red, reference)**: lower right ribs.

Keep the leads out of the arm's swing and fix the cable to the vest every 10
cm; cable movement is the largest source of noise. Gel electrodes give the
cleanest signal for trials. For repeated matches, use a chest strap with
textile electrodes, moistened before play, under a snug compression vest.

### What the software does with it

The firmware samples at 250 Hz from its own task, so Wi-Fi never delays a
sample, and sends 50 samples per packet. The hub detects beats
(Pan-Tompkins), drops beat intervals that are physiologically implausible,
and reports heart rate. In still windows of 30 s or more it also reports RMSSD,
the standard short-term measure of heart-rate variability used for recovery.
When an electrode comes off, the AD8232 raises LO+ or LO-, and the vest page
says "electrode off" instead of showing a number.

## 6. NEO-6M GPS

- VCC to 3V3 (the module has its own regulator and accepts 3.3 V).
- Mount the ceramic antenna **face up**, on top of the pod, with nothing metal
  above it. Keep it at least 3 cm from the ESP32's antenna.
- The firmware asks for 5 fixes per second with only the two sentences it uses.
  The board's EEPROM and backup cell keep satellite data between sessions, so
  the next start finds a fix in seconds instead of half a minute. Switch the vest
  on outdoors a minute before the warm-up.
- 5 Hz is the most the NEO-6M can do. For research-grade speed, a 10 Hz
  multi-constellation receiver (u-blox M10 class) is the upgrade path.

## 7. What the BMP180 could and could not do

Altitude from air pressure moves about 0.25 m with sensor noise at its best
setting, while a football pitch is flat, so it adds nothing for position. It
could, in principle, mark jumps, but the accelerometer already sees take-off
and landing much more precisely. It is left unread to save I2C time.

## 8. Build order and checks

1. **Power only.** Connect the battery, TP4056, switch and TPS63020 with nothing
   else. Switch on: VOUT should read 3.28 to 3.35 V. Switch off: 0 V.
2. **ESP32.** Wire 3V3 and GND with the capacitors. Flash from *Flash a vest*
   (USB plugged in, battery switched off). The sketch needs the esp32 core 3.x
   and four libraries from the Arduino library manager: Adafruit MPU6050 (with
   Adafruit Unified Sensor and BusIO), ArduinoJson 7 and TinyGPSPlus. The
   serial monitor should print `CPG44 ESP32-S3 VEST: ECG, IMU, GNSS`.
3. **GY-87.** The serial monitor lists `GY-87 MPU6050 0x68: FOUND`,
   `GY-87 BMP180 0x77: FOUND`, `MPU6050 ready` and the magnetometer type.
4. **GPS.** `NEO-6M UART started`. Outdoors, the vest page shows satellites
   within a minute and "GNSS fixes" close to 100 %.
5. **ECG.** Put on three gel electrodes. The vest page shows a heart rate within
   10 seconds; rub an electrode lead and the number should hold or say "noisy",
   never jump.
6. **Battery.** The vest page shows a percentage. Compare the voltage with a
   multimeter at the battery: within 0.05 V is right.
7. **Full run.** With all modules on, the *Samples arriving* bars on the vest
   page should sit near 100 %.

## 9. Calibration before each session

| Sensor | What to do | Time |
|---|---|---|
| Gyroscope | Lay the vest flat and still | 2 s |
| Accelerometer | Rest the pod still on each of its six faces | 15 s, once per pod |
| Magnetometer | Turn the pod slowly through a figure eight, away from metal | 20 s |
| ECG | Check the heart rate at rest against your pulse at the wrist (count 30 s, double it) | 30 s |

The accelerometer and gyroscope results are saved by the hub, so the six faces
are needed only once per pod. Every number on the vest page carries its own
state: calibrated, waiting, or why it is missing.

## 10. Safety

- The AD8232 is a monitoring front end, not a medical device. Never use it with
  the vest plugged into a charger or a laptop.
- Use only a LiPo with its own protection board, in a padded pocket on the back.
  Replace any cell that swells.
- No hard edges facing the body. The pod goes between the shoulder blades,
  where players rarely land, and where vest-worn GNSS units sit in
  professional football.
