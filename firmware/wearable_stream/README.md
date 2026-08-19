Flash this sketch on the ESP32-S3. Set Wi-Fi at the top of `wearable_stream.ino`.

The board listens on TCP port 9000. In WSL:

    python -m soccer_analytics.hub --esp32 <ESP32_IP>
