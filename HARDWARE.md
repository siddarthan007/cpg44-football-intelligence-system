# Wearable hardware and firmware

Use `firmware/wearable_stream/wearable_stream.ino`. This is the sensor-tested
sketch and the supported field firmware. It publishes only to
`https://cpg44.nivaspms.com/api/v1/sensors/ingest`.

## Connections

The full step-by-step build, shopping list and checks are in
[docs/hardware/BUILD_GUIDE.md](docs/hardware/BUILD_GUIDE.md), with the wiring
diagram in [docs/hardware/vest_wiring.svg](docs/hardware/vest_wiring.svg).
`python scripts/check_vest_circuit.py` checks the circuit against the
datasheets and the firmware pins. Pin summary:

| Connection | ESP32-S3 pin | Setting |
|---|---:|---|
| GY-87 SDA (MPU6050, magnetometer) | GPIO 4 | I2C, 100 kHz |
| GY-87 SCL | GPIO 5 | I2C, 100 kHz |
| NEO-6M TX to ESP32 RX | GPIO 16 | 9600 baud |
| ESP32 TX to NEO-6M RX (through 1 kOhm) | GPIO 17 | sets 5 Hz at boot |
| AD8232 OUTPUT | GPIO 2 | ADC1, 250 Hz |
| AD8232 LO+ / LO- | GPIO 11 / 12 | electrode-off |
| AD8232 3.3v and SDN | 3.3 V through 10 Ohm, 10 uF to GND | supply filter; SDN high keeps it on |
| Battery divider (2 x 100 kOhm) | GPIO 1 | ADC1 |
| TPS63020 3.3 V output | 3V3 pin, through a 2-pin power plug | unplug it before connecting USB |

The BMP180 on the GY-87 (0x77) is detected for the build check and not
streamed. There is no pulse oximeter and no fuel gauge: heart rate comes from
the AD8232 chest ECG and battery charge from the divider on GPIO 1.

## Sampling

| Sensor | Output |
|---|---:|
| MPU6050 | 50 Hz, ±16 g, ±2000 °/s, on-chip low-pass 21 Hz |
| Magnetometer (HMC5883L or QMC5883L) | 50 Hz, with each IMU sample |
| AD8232 ECG | 250 Hz, sent in blocks of 50 |
| NEO-6M GPS | 5 Hz, RMC and GGA only |
| Battery | every 2 s |

The low-pass must stay below the 25 Hz Nyquist limit of 50 Hz sampling, and
±16 g avoids clipping on kicks and tackles. Acceleration is sent in m/s²,
angular rate in rad/s and the magnetic field in microtesla. Heart rate is
computed on the PC from the raw ECG so electrode contact and noise can be
checked before a number is shown.

## Flash from WSL

Set these before starting the backend:

```bash
export CPG44_RELAY_TOKEN="<same token as the VPS>"
export CPG44_RELAY_CA_FILE="$PWD/configs/relay_ca.pem"
```

Attach the board in an elevated Windows PowerShell:

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Open Dashboard > ESP32 setup. Select the port, enter the player ID, match ID,
SSID and password, then flash. The relay token and Wi-Fi values are written to
the ignored `generated_config.h` only.

Manual build check:

```bash
conda activate soccer
arduino-cli compile --fqbn esp32:esp32:esp32s3 firmware/wearable_stream
arduino-cli upload --port /dev/ttyACM0 \
  --fqbn esp32:esp32:esp32s3 firmware/wearable_stream
```

## Placement

The wearable design (compression vest, back pod between the shoulder blades,
three ECG electrodes on the chest, harness, enclosure, power budget, parts list
and bench tests) is in [docs/hardware/BUILD_GUIDE.md](docs/hardware/BUILD_GUIDE.md).
Replaying 21 bench sessions from the first prototype
(`python -m soccer_analytics.sensors.session_report`) showed three field risks:
samples lost on the way to the laptop, no GNSS fix indoors, and an optical pulse
sensor that was rarely on the skin. The first was fixed in the firmware and
relay (all samples now arrive, about a quarter of a second after they are
taken); the third by replacing the optical sensor with a chest ECG.


- Fix the IMU firmly in the same upper-back position for each session.
- Fix the ECG leads to the vest every 10 cm; cable movement is the main source of noise.
- Keep the GPS patch facing upward.
- Use strain relief, an insulated enclosure and a protected battery supply.

Validate every assembled unit using
`docs/CAPSTONE_ACCURACY_PROTOCOL.md`. Do not use the module data sheet as the
accuracy result for the complete wearable.
