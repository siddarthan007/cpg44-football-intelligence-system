"""The vest's serial self-test becomes a checklist the setup page can show."""

from cpg44_api.flasher import parse_check_line


def states(line):
    return {r["key"]: r["state"] for r in parse_check_line(line)}


def test_healthy_vest_is_all_green():
    line = ("[check] i2c 2C 68 77 | imu 9.8 m/s2 at 50 Hz | mag QMC5883P 45 uT | gps 9000 chars, 8 sats, fix | "
            "ecg 1400-2700, electrodes on | battery 3.95 V 82 % | wifi EACCESS-M1 -60 dBm | relay sent 5000")
    assert set(states(line).values()) == {"ok"}


def test_bench_faults_are_named_with_a_fix():
    line = ("[check] i2c none | imu missing | mag none | gps nothing on GPIO16 or GPIO17: check VCC, GND, TX | "
            "ecg 0-2099, electrodes off | battery not wired (GPIO1) | wifi not connected | relay sent 0")
    rows = {r["key"]: r for r in parse_check_line(line)}
    assert rows["bus"]["state"] == "bad" and "GPIO 4" in rows["bus"]["hint"]
    assert rows["gps"]["state"] == "bad" and "GPIO 16" in rows["gps"]["hint"]
    assert rows["ecg"]["state"] == "warn" and rows["battery"]["state"] == "warn"
    assert rows["wifi"]["state"] == "bad" and "2.4 GHz" in rows["wifi"]["hint"]
    assert rows["relay"]["state"] == "bad"


def test_slow_motion_sampling_and_flat_ecg_are_flagged():
    line = ("[check] i2c 68 77 | imu 9.4 m/s2 at 21 Hz, loop paused 120 ms | mag none | gps 10 chars, 0 sats, no fix | "
            "ecg 2048-2050, electrodes on | battery 3.40 V 8 % | wifi LC -85 dBm | relay sent 10")
    s = states(line)
    assert s["imu"] == "warn" and s["ecg"] == "bad" and s["battery"] == "warn" and s["wifi"] == "warn" and s["gps"] == "warn"


def _gps_row(text):
    from cpg44_api.flasher import parse_check_line
    rows = parse_check_line(f"[check] i2c 68 | imu 9.8 m/s2 at 50 Hz | mag none | gps {text} | ecg 1500-2600, electrodes on "
                            f"| battery 0 V on GPIO1 | wifi x -60 dBm | relay sent 1 | chip 47 C, tx 15.0 dBm, 160 MHz")
    return next(r for r in rows if r["key"] == "gps")


def test_gps_line_probe_states_have_their_own_advice():
    assert _gps_row("TX held low: GPS has GND but no power, check VCC")["value"] == "no power"
    assert _gps_row("GPS powered but silent: module not sending")["value"] == "powered, sends nothing"
    assert _gps_row("nothing connected on GPIO16 or GPIO17: check TX wire")["value"] == "TX wire open"
    assert _gps_row("seen at times: loose TX wire or contact")["value"] == "comes and goes"
    assert _gps_row("found on GPIO43 instead of GPIO16, no valid sentence yet")["state"] == "warn"


def test_usbipd_ignores_the_linux_usbip_daemon(monkeypatch):
    from cpg44_api import usb_bridge
    monkeypatch.setattr(usb_bridge.shutil, "which", lambda name: "/usr/bin/usbipd" if name == "usbipd" else None)
    monkeypatch.setattr(usb_bridge, "Path_exists", lambda path: False)
    assert usb_bridge._usbipd_exe() is None
