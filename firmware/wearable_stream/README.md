# ESP32-S3 relay wearable

This is the supported sensor-tested firmware. It reads MPU6050, MAX30102 and
NEO-6M data and sends timestamped raw batches only to:

```text
https://cpg44.nivaspms.com/api/v1/sensors/ingest
```

Use Dashboard > ESP32 setup to flash player ID, match ID, Wi-Fi settings, relay
token and the verified relay CA chain. These values are written to the ignored
`generated_config.h`.

The HTTPS task runs separately from sensor sampling. Failed sends keep a small
bounded RAM queue. The VPS relay provides the longer few-megabyte reconnect
window.

## Sampling used by the tested sketch

| Sensor | Setting | Reason |
|---|---:|---|
| MPU6050 accelerometer | 100 Hz, +/-8 g | Keeps running and impact peaks inside range |
| MPU6050 gyroscope | 100 Hz, +/-1000 degrees/s | Allows fast turns without using the widest range |
| MPU6050 low-pass filter | 44 Hz | Fits the 100 Hz sample stream |
| MAX30102 ADC | 100 samples/s, average 4 | Gives a 25 Hz FIFO output for the host filter |
| MAX30102 pulse | 411 microseconds, 8192 nA range | Uses the high-resolution pulse setting with moderate headroom |

The ESP32 sends raw optical samples. BPM and SpO2 are calculated on the PC.
They are left empty when contact, motion, estimator agreement, or optical
quality checks fail. SpO2 remains an uncalibrated estimate.

The clock uses SNTP smooth updates and anchors each acquisition timestamp to
`esp_timer`. The anchor is refreshed every minute. The relay keeps the source
timestamp unchanged.

Primary references:

- [Analog Devices MAX30102 product page and Rev. 1 data sheet](https://www.analog.com/en/products/MAX30102.html)
- [TDK MPU-6000 and MPU-6050 register map](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)
- [Espressif system time and SNTP guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/system_time.html)
- [SparkFun MAX3010x driver](https://github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library)

Build check:

```bash
conda run -n soccer arduino-cli compile \
  --build-path /tmp/cpg44-wearable-build \
  --fqbn esp32:esp32:esp32s3 firmware/wearable_stream
```
