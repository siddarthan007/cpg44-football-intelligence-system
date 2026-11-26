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
[check] i2c 2C 68 77 | imu 9.4 m/s2 at 50 Hz | mag QMC5883P 31 uT | gps nothing connected on GPIO16 or GPIO17: check TX wire | ecg 364-1980, electrodes off | battery 0 V on GPIO1 | wifi EACCESS-M1 -70 dBm | relay sent 555 | chip 47 C, tx 17.0 dBm, 160 MHz
```

It also finds a few wiring faults by itself:

- **GPS.** Before the serial port takes the GPS pins, the vest tests each line
  electrically: with its own pull-down and pull-up, and by counting edges for a
  second. That separates four cases a serial port cannot: the module is
  sending, it is powered but silent, it has ground but no power (the line is
  held low), or nothing is connected (the line just follows the pull). If
  neither GPIO 16 nor 17 carries a signal, it listens on every free pin at once
  and reads the GPS wherever its TX wire actually is. The baud rate is measured
  from the shortest pulse on the line instead of guessed. While no sentence
  arrives it checks again every 20 s, so a loose wire recovers by itself and
  shows up as "seen at times".
- **Battery.** 0 V on GPIO 1 means the battery switch is off (normal on USB
  power) or the divider is not wired.
- **Wi-Fi.** A network out of range: it lists the networks it can see.

## Heat

The chip temperature, transmit power and CPU clock are on every self-test line
and in the status the vest sends. The CPU runs at 160 MHz (sampling and
compression use a fraction of it), and transmit power follows the access
point's signal: 11 dBm on a strong link, up to 19.5 dBm on a weak one or when
deliveries start failing. The radio itself stays awake, because modem sleep
would delay every relay reply by up to a beacon interval. On the bench the chip
reads 47 to 48 °C while streaming, normal for an ESP32-S3 on Wi-Fi (rated to
85 °C); the metal can feels hot to the touch at that temperature.

## Delivery

Up to 288 samples per POST, deflated with the compressor in the chip's ROM.
Samples wait up to 15 s in a 1024-sample queue in PSRAM, so a slow network
patch delays data instead of losing it. Against an older relay (100 per POST,
no compressed bodies) the vest falls back to 96 plain samples at a time, and
offers the larger, compressed batches again after 10 minutes. With nothing
accepted for 90 s while on Wi-Fi it rebuilds the HTTPS connection, and after
5 minutes it restarts.

## Build check

```bash
conda run -n soccer arduino-cli compile \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi" \
  firmware/wearable_stream
```

With arduino-esp32 3.3.11 this uses 1,110,915 bytes of flash (35 %) and
about 73 KB of static RAM.

## References

- [Analog Devices AD8232 data sheet](https://www.analog.com/en/products/ad8232.html)
- [TDK MPU-6000 and MPU-6050 register map](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)
- [u-blox 6 receiver description (UBX CFG-RATE, CFG-MSG)](https://www.u-blox.com/en/docs/UBX-13003221)
- [Espressif system time and SNTP guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/system_time.html)
