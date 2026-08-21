"""CLI entry point for the relay-backed wearable processor."""

from soccer_analytics.sensors.hub import main

if __name__ == "__main__":
    raise SystemExit(main() or 0)
