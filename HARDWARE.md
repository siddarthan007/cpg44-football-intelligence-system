# Wearable hardware and firmware

Use `firmware/wearable_stream/wearable_stream.ino`. This is the sensor-tested
sketch and the supported field firmware. It publishes only to
`https://cpg44.nivaspms.com/api/v1/sensors/ingest`.

## Connections

| Connection | ESP32-S3 pin | Setting |
|---|---:|---|
| MPU6050 and MAX30102 SDA | GPIO 4 | I2C, 100 kHz |
| MPU6050 and MAX30102 SCL | GPIO 5 | I2C, 100 kHz |
| NEO-6M TX to ESP32 RX | GPIO 16 | 9600 baud |
| ESP32 TX to NEO-6M RX | GPIO 17 | optional |

All boards must share ground. Check the voltage requirement printed on each
breakout before wiring. Expected I2C addresses are `0x57` for MAX30102 and
`0x68` for MPU6050.

## Sampling

| Sensor | Current output |
|---|---:|
| MPU6050 | about 100 Hz |
| MAX30102 red and IR FIFO | 25 Hz |
| NEO-6M GPS | 1 Hz |

Acceleration is sent in m/s² and angular rate in rad/s. Raw PPG is processed on
the PC so motion, contact and optical quality can be checked before BPM or SpO2
is shown.

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

- Fix the IMU firmly in the same upper-back position for each session.
- Keep the MAX30102 against skin and shield it from outside light.
- Keep the GPS patch facing upward.
- Use strain relief, an insulated enclosure and a protected battery supply.

Validate every assembled unit using
`docs/CAPSTONE_ACCURACY_PROTOCOL.md`. Do not use the module data sheet as the
accuracy result for the complete wearable.
