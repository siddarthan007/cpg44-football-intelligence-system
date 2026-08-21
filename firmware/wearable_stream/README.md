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

Build check:

```bash
conda run -n soccer arduino-cli compile \
  --build-path /tmp/cpg44-wearable-build \
  --fqbn esp32:esp32:esp32s3 firmware/wearable_stream
```
