# ESP32-S3 vest firmware

Reads the GY-87 (MPU6050 motion, HMC5883L or QMC5883L magnetometer), the
AD8232 chest ECG, the NEO-6M GNSS receiver and the battery voltage, and sends
timestamped raw batches only to:

```text
https://cpg44.nivaspms.com/api/v1/sensors/ingest
```

Use Flash a vest in the dashboard to write the player number, match ID, Wi-Fi
settings, relay token and the verified relay CA chain. These values go into the
ignored `generated_config.h`; a build without it stays offline.

Sampling runs apart from sending: the ECG has its own FreeRTOS task, the HTTPS
sender another, and a bounded RAM queue covers short Wi-Fi drops. The relay
keeps a few megabytes for longer reconnects.

## What it sends

| Stream | Rate | Settings |
|---|---:|---|
| Motion (MPU6050) | 50 Hz | ±16 g, ±2000 °/s, on-chip low-pass 21 Hz (below the 25 Hz Nyquist limit) |
| Magnetometer | with each motion sample | bypass mode through the MPU6050, µT |
| ECG (AD8232) | 250 Hz in blocks of 50 | 12-bit ADC1 counts, electrode-off flag from LO+ and LO- |
| GNSS (NEO-6M) | 5 Hz | RMC and GGA only, set over UBX at every boot |
| Status | every 2 s | Wi-Fi strength, battery voltage and charge, delivered rates, dropped and queued samples |

Heart rate, heart-rate variability and electrode quality are worked out on the
laptop (`soccer_analytics/sensors/ecg.py`), where a whole window of signal can
be checked before a number is shown.

The clock is set by an NTP query over a plain UDP socket once Wi-Fi is up
(every 10 minutes; the relay's HTTP Date header is the fallback) and every
acquisition timestamp is anchored to `esp_timer`. The core's built-in SNTP
client is not used: when its first DNS lookup failed before Wi-Fi connected,
its retry ran outside the network-stack lock and rebooted the board.

## Bench self-test

Every 2 s the vest prints one line on USB, which the app's Flash a vest page
turns into a checklist:

```text
[check] i2c 2C 68 77 | imu 9.4 m/s2 at 50 Hz | mag QMC5883P 25 uT | gps nothing on GPIO16 or GPIO17: check VCC, GND, TX | ecg 0-2099, electrodes off | battery not wired (GPIO1) | wifi EACCESS-M1 -61 dBm | relay sent 862
```

It also finds a few wiring faults by itself: GPS TX and RX swapped (it
listens on the other pin), a GPS module set to another baud rate (it tries
9600, 38400 and 115200), a battery divider that is not wired (it reports no
level instead of reading a floating pin) and a Wi-Fi network out of range (it
lists the networks it can see). Batches to the relay are deflated with the
compressor in the chip's ROM when the relay accepts it, and sent as plain JSON
otherwise.

## Build check

```bash
conda run -n soccer arduino-cli compile \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi" \
  firmware/wearable_stream
```

With arduino-esp32 3.3.12 this uses 1,081,499 bytes of flash (34 %) and
72,796 bytes of static RAM (22 %).

## References

- [Analog Devices AD8232 data sheet](https://www.analog.com/en/products/ad8232.html)
- [TDK MPU-6000 and MPU-6050 register map](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)
- [u-blox 6 receiver description (UBX CFG-RATE, CFG-MSG)](https://www.u-blox.com/en/docs/UBX-13003221)
- [Espressif system time and SNTP guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/system_time.html)
