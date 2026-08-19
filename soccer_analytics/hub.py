"""CLI: python -m soccer_analytics.hub --esp32 192.168.x.x"""

from soccer_analytics.sensors.hub import main

if __name__ == "__main__":
    raise SystemExit(main() or 0)
