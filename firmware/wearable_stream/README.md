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

The clock uses SNTP and anchors every acquisition timestamp to `esp_timer`;
the anchor is refreshed every minute and the relay keeps the source timestamp
unchanged.

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
