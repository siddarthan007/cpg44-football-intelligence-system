"""ESP32 provisioning backed by the real Arduino CLI and serial devices."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[3]
FIRMWARE_DIR = ROOT / "firmware" / "wearable_stream"
RELAY_ENDPOINT = "https://cpg44.nivaspms.com/api/v1/sensors/ingest"
MATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")



# ESP32-S3-WROOM-1 N16R8: 16 MB quad flash and 8 MB octal PSRAM. The default
# FQBN assumes 4 MB flash with no PSRAM; this uses the whole flash (3 MB app)
# and enables the octal PSRAM so larger relay buffers have room.
FQBN = "esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi"


def _arduino_cli():
    """arduino-cli on PATH, or installed beside this Python (conda environment)."""
    found = shutil.which("arduino-cli")
    if found:
        return found
    local = Path(sys.executable).with_name("arduino-cli")
    return str(local) if local.is_file() else None

class ESP32Flasher:
    def __init__(
        self,
        firmware_dir: Path = FIRMWARE_DIR,
        relay_token: Optional[str] = None,
        relay_ca_pem: Optional[str] = None,
    ):
        self.firmware_dir = firmware_dir
        self.relay_token = (
            relay_token if relay_token is not None
            else os.environ.get("CPG44_RELAY_TOKEN", "").strip()
        )
        self.relay_ca_pem = (
            relay_ca_pem if relay_ca_pem is not None
            else self._load_relay_ca(os.environ.get("CPG44_RELAY_CA_FILE", ""))
        )
        self.active_flash_job: Optional[dict] = None
        self.lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        # Serial monitor state: one background reader per port, ring buffer of
        # decoded lines that the dashboard polls.
        self.monitor_port: Optional[str] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_stop = threading.Event()
        self.monitor_lines: List[dict] = []
        self.monitor_lock = threading.RLock()

    # ---- serial monitor -------------------------------------------------- #
    MONITOR_MAX_LINES = 400

    def start_monitor(self, port: str) -> dict:
        """Start tailing a serial port for the dashboard monitor."""
        if not self._port_exists(port):
            return {"ok": False, "error": f"serial port is not attached: {port}"}
        with self.monitor_lock:
            if self.monitor_thread is not None and self.monitor_thread.is_alive() \
                    and self.monitor_port == port:
                return {"ok": True, "port": port, "already_running": True}
            self.stop_monitor()
            self.monitor_stop.clear()
            self.monitor_lines = []
            self.monitor_port = port
            self.monitor_thread = threading.Thread(
                target=self._monitor_worker, args=(port,), daemon=True,
            )
            self.monitor_thread.start()
        return {"ok": True, "port": port}

    def stop_monitor(self) -> dict:
        with self.monitor_lock:
            self.monitor_stop.set()
            thread = self.monitor_thread
            self.monitor_thread = None
            self.monitor_port = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        return {"ok": True}

    def monitor_status(self) -> dict:
        with self.monitor_lock:
            lines = list(self.monitor_lines[-self.MONITOR_MAX_LINES:])
            running = bool(self.monitor_thread is not None and self.monitor_thread.is_alive())
            port = self.monitor_port
        check = next((row for row in reversed(lines) if row["text"].startswith("[check]")), None)
        notes = [row["text"] for row in lines if row["text"].startswith(("Wi-Fi:", "GPS:", "Battery divider", "Clock set", "Relay does", "Relay takes", "assert failed"))]
        crashed = any(row["text"].startswith(("assert failed", "Guru Meditation", "Backtrace")) for row in lines[-120:])
        return {
            "port": port,
            "running": running,
            "lines": lines,
            "check": parse_check_line(check["text"]) if check else None,
            "check_age_s": round(time.time() - check["t"], 1) if check else None,
            "notes": notes[-6:],
            "crashed": crashed,
        }

    def _monitor_worker(self, port: str) -> None:
        try:
            import serial as pyserial
        except ImportError:
            self._append_monitor_line({"t": time.time(), "text": "pyserial is not installed"})
            return
        try:
            # Set the control lines before opening: opening with them in their
            # default state and changing them afterwards resets the ESP32-S3
            # (its USB serial treats that toggle as a reset request), so just
            # watching the vest rebooted it.
            ser = pyserial.Serial()
            ser.port, ser.baudrate, ser.timeout = port, 115200, 0.5
            ser.dtr = False
            ser.rts = False
            ser.open()
        except Exception as exc:
            self._append_monitor_line({
                "t": time.time(), "text": f"could not open {port}: {exc}",
            })
            return
        try:
            buffer = b""
            while not self.monitor_stop.is_set():
                try:
                    n = ser.in_waiting
                    chunk = ser.read(n if n else 1)
                except Exception:
                    break
                if not chunk:
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    raw, _, rest = buffer.partition(b"\n")
                    buffer = rest
                    text = raw.decode("utf-8", "replace").rstrip("\r")
                    if text:
                        self._append_monitor_line({"t": time.time(), "text": text})
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def _append_monitor_line(self, row: dict) -> None:
        with self.monitor_lock:
            self.monitor_lines.append(row)
            if len(self.monitor_lines) > self.MONITOR_MAX_LINES * 2:
                del self.monitor_lines[:-self.MONITOR_MAX_LINES]

    @staticmethod
    def _load_relay_ca(path_value: str) -> str:
        path_text = str(path_value or "").strip()
        if not path_text:
            return ""
        path = Path(path_text).expanduser()
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def relay_configuration_status(self) -> dict:
        token_ready = len(self.relay_token) >= 32
        ca_ready = (
            "-----BEGIN CERTIFICATE-----" in self.relay_ca_pem
            and "-----END CERTIFICATE-----" in self.relay_ca_pem
        )
        return {
            "endpoint": RELAY_ENDPOINT,
            "token_configured": token_ready,
            "ca_configured": ca_ready,
            "ready": token_ready and ca_ready,
        }

    def list_ports(self) -> List[dict]:
        """List real serial ports only.

        Inside WSL the kernel also exposes the motherboard's legacy 8250 UARTs
        as /dev/ttyS0..15 with no hardware behind them, which used to fill the
        flasher page with dead entries. Those are filtered out. When WSL has no
        USB pass-through but Windows shows the ESP32 on a COM port, the Windows
        port is listed as a hint so the operator knows to run the usbipd
        attach command instead of seeing an empty page."""
        ports = []
        try:
            import serial.tools.list_ports
            for port in serial.tools.list_ports.comports():
                device = port.device or ""
                hwid = (port.hwid or "")
                # Drop legacy ISA UART placeholders: no USB identity, no
                # description, and a ttyS name.
                if device.startswith("/dev/ttyS") and hwid in ("", "n/a"):
                    continue
                ports.append({
                    "device": device,
                    "name": port.name,
                    "description": port.description,
                    "hwid": hwid,
                    "is_usb": "USB" in hwid.upper()
                    or "UART" in (port.description or "").upper(),
                })
        except Exception:
            return []
        if not ports:
            windows_hint = self._windows_com_hint()
            ports.extend(windows_hint)
        return ports

    @staticmethod
    def _windows_com_hint() -> List[dict]:
        """Ask Windows for ESP32 COM ports when WSL itself has none attached."""
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "Get-PnpDevice -Class Ports -PresentOnly | "
                 "Select-Object -ExpandProperty FriendlyName"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return []
        rows = []
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            # Match ESP32-family adapters by name; usbipd shows them on COMx.
            if "COM" not in line:
                continue
            if not any(tag in line.lower() for tag in
                       ("esp32", "jtag", "serial", "cp210", "ch340", "ftdi")):
                continue
            import re as _re
            match = _re.search(r"\(COM(\d+)\)", line)
            if not match:
                continue
            com = f"COM{match.group(1)}"
            rows.append({
                "device": com,
                "name": com.lower(),
                "description": f"{line} (Windows host, not attached to WSL)",
                "hwid": "windows-host",
                "is_usb": True,
                "windows_only": True,
            })
        return rows

    def last_flash_settings(self) -> dict:
        """Player, match and Wi-Fi name of the last build (never the password)."""
        header = self.firmware_dir / "generated_config.h"
        try:
            text = header.read_text(encoding="utf-8")
        except OSError:
            return {}
        out = {}
        for key, pattern in (("player_id", r"PLAYER_ID = (\d+);"), ("match_id", r'MATCH_ID\[\] = (".*?");'),
                             ("wifi_ssid", r'WIFI_SSID\[\] = (".*?");')):
            m = re.search(pattern, text)
            if m:
                out[key] = int(m.group(1)) if key == "player_id" else json.loads(m.group(1))
        return out

    def toolchain_status(self) -> dict:
        arduino_cli = _arduino_cli()
        esptool_available = False
        try:
            result = subprocess.run(
                [sys.executable, "-m", "esptool", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            esptool_available = result.returncode == 0
        except Exception:
            pass
        ports = self.list_ports()
        relay = self.relay_configuration_status()
        return {
            "ready": bool(arduino_cli and ports and relay["ready"]),
            "arduino_cli": arduino_cli,
            "esptool_available": esptool_available,
            "serial_ports": ports,
            "wsl_usb_required": not bool(ports),
            "relay": relay,
            "last_flash": self.last_flash_settings(),
            "message": (
                "Ready to compile and flash the relay-only firmware."
                if arduino_cli and ports and relay["ready"]
                else "Attach the ESP32, install Arduino CLI, and configure the relay token and CA file."
            ),
        }

    def _port_exists(self, port: str) -> bool:
        return any(item["device"] == port for item in self.list_ports())

    def get_chip_info(self, port: str) -> dict:
        """Identify the board from its USB identity, without touching it.

        The old probe ran ``esptool chip-id``, which resets the ESP32 into its
        bootloader: merely opening the setup page rebooted a running vest and
        fought the serial monitor for the port."""
        match = next((item for item in self.list_ports() if item["device"] == port), None)
        if match is None:
            return {"ok": False, "port": port, "error": "serial port is not attached to WSL"}
        hwid = (match.get("hwid") or "").upper()
        espressif = "VID:PID=303A" in hwid
        return {"ok": True, "port": port, "espressif_usb": espressif,
                "label": "ESP32-S3 (USB serial)" if espressif else (match.get("description") or port)}

    def generate_config_header(
        self,
        player_id: int,
        wifi_ssid: str,
        wifi_pass: str,
        endpoint: str = "",
        send_rate_hz: int = 10,
        match_id: str = "live",
    ) -> str:
        """Create the ignored build-time credential header consumed by the sketch."""
        del send_rate_hz
        if not 1 <= int(player_id) <= 9999:
            raise ValueError("player_id must be between 1 and 9999")
        match_id = str(match_id or "live").strip()
        if not MATCH_ID_PATTERN.fullmatch(match_id):
            raise ValueError("match_id may contain 1 to 64 letters, digits, dots, colons, underscores or hyphens")
        if not wifi_ssid or len(wifi_ssid.encode("utf-8")) > 32:
            raise ValueError("Wi-Fi SSID must contain 1 to 32 bytes")
        password_size = len(wifi_pass.encode("utf-8"))
        if wifi_pass and not 8 <= password_size <= 63:
            raise ValueError("Wi-Fi password must contain 8 to 63 bytes, or be blank for an open network")
        if endpoint and endpoint != RELAY_ENDPOINT:
            raise ValueError(f"relay endpoint is fixed to {RELAY_ENDPOINT}")
        # Re-read the certificate file so a refreshed one (scripts/prepare_relay_ca.sh)
        # is used without restarting the app.
        fresh_ca = self._load_relay_ca(os.environ.get("CPG44_RELAY_CA_FILE", ""))
        if fresh_ca:
            self.relay_ca_pem = fresh_ca
        relay = self.relay_configuration_status()
        if not relay["token_configured"]:
            raise ValueError("CPG44_RELAY_TOKEN must contain at least 32 characters")
        if not relay["ca_configured"]:
            raise ValueError("CPG44_RELAY_CA_FILE must contain the CA certificates trusted for the Cloudflare edge")
        content = (
            "// Generated locally by the CPG44 provisioning screen. Do not commit.\n"
            "#pragma once\n\n"
            f"inline constexpr int PLAYER_ID = {int(player_id)};\n"
            f"inline constexpr char MATCH_ID[] = {json.dumps(match_id)};\n"
            f"inline constexpr char WIFI_SSID[] = {json.dumps(wifi_ssid)};\n"
            f"inline constexpr char WIFI_PASS[] = {json.dumps(wifi_pass)};\n"
            f"inline constexpr char RELAY_ENDPOINT[] = {json.dumps(RELAY_ENDPOINT)};\n"
            f"inline constexpr char RELAY_TOKEN[] = {json.dumps(self.relay_token)};\n"
            f"inline constexpr char RELAY_CA_PEM[] = {json.dumps(self.relay_ca_pem)};\n"
        )
        self.firmware_dir.mkdir(parents=True, exist_ok=True)
        header_path = self.firmware_dir / "generated_config.h"
        header_path.write_text(content, encoding="utf-8")
        return str(header_path)

    def flash_device(
        self,
        port: str,
        player_id: int,
        wifi_ssid: str,
        wifi_pass: str,
        endpoint: str = "",
        baud_rate: int = 921600,
        match_id: str = "live",
    ) -> dict:
        """Compile and upload the working HTTPS relay wearable asynchronously."""
        arduino_cli = _arduino_cli()
        if not arduino_cli:
            return {"ok": False, "error": "arduino-cli is not installed in WSL"}
        if not self._port_exists(port):
            return {"ok": False, "error": f"serial port is not attached: {port}"}
        try:
            header = self.generate_config_header(
                player_id, wifi_ssid, wifi_pass, endpoint, match_id=match_id
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        job_id = f"flash_{int(time.time())}"
        del baud_rate  # use the board core's safe default upload speed
        with self.lock:
            if self._process and self._process.poll() is None:
                return {"ok": False, "error": "a flash process is already running"}
            self.active_flash_job = {
                "id": job_id,
                "port": port,
                "player_id": int(player_id),
                "status": "starting",
                "progress_pct": None,
                "logs": ["Preparing Arduino build and upload…"],
                "started_at": time.time(),
                "config_header": header,
                "pid": None,
            }
        threading.Thread(
            target=self._run_flash_worker,
            args=(job_id, arduino_cli, port),
            daemon=True,
        ).start()
        return {"ok": True, "job": copy.deepcopy(self.active_flash_job)}

    def _run_flash_worker(self, job_id: str, arduino_cli: str, port: str):
        try:
            upload_command = [
                arduino_cli, "upload", "--port", port,
                "--fqbn", FQBN, str(self.firmware_dir),
            ]
            # Arduino CLI documents compile and upload as separate commands.
            # Keeping them separate produces an accurate failure stage and avoids
            # relying on board-core-specific upload-property flags. CDCOnBoot=cdc
            # routes the app's Serial prints to the same USB port used for
            # flashing, so the dashboard serial monitor sees the boot log.
            compile_command = [
                arduino_cli, "compile", "--fqbn", FQBN,
                str(self.firmware_dir),
            ]
            return_code = 0
            resume_monitor = None
            for stage, stage_command in (("compile", compile_command), ("upload", upload_command)):
                if stage == "upload" and self.monitor_port:
                    # the monitor holds the port open; the upload needs it
                    resume_monitor = self.monitor_port
                    self.stop_monitor()
                with self.lock:
                    self.active_flash_job["logs"].append(f"Starting {stage}…")
                    self.active_flash_job["status"] = stage
                process = subprocess.Popen(
                    stage_command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self.lock:
                    self._process = process
                    self.active_flash_job["pid"] = process.pid
                assert process.stdout is not None
                for line in process.stdout:
                    message = line.strip()
                    if not message:
                        continue
                    with self.lock:
                        if self.active_flash_job and self.active_flash_job.get("id") == job_id:
                            self.active_flash_job["logs"].append(message)
                            self.active_flash_job["logs"] = self.active_flash_job["logs"][-200:]
                return_code = process.wait()
                if return_code != 0:
                    break
            with self.lock:
                if self.active_flash_job and self.active_flash_job.get("id") == job_id:
                    self.active_flash_job.update({
                        "status": "completed" if return_code == 0 else "failed",
                        "return_code": return_code,
                        "completed_at": time.time(),
                        "progress_pct": 100 if return_code == 0 else None,
                    })
                self._process = None
            # Watch the fresh boot: reopen the monitor on the vest's port.
            if return_code == 0 or resume_monitor:
                time.sleep(1.5)
                self.start_monitor(resume_monitor or port)
        except Exception as exc:
            with self.lock:
                if self.active_flash_job and self.active_flash_job.get("id") == job_id:
                    self.active_flash_job.update({
                        "status": "failed",
                        "error": str(exc),
                        "completed_at": time.time(),
                    })
                self._process = None

    def get_flash_status(self) -> dict:
        with self.lock:
            return {"active_job": copy.deepcopy(self.active_flash_job)}


# ---- the vest's serial self-test, as a checklist ----------------------------
# The firmware prints one line every 2 s, for example:
# [check] i2c 2C 68 77 | imu 9.4 m/s2 at 50 Hz | mag QMC5883P 25 uT | gps 1234 chars, 7 sats, fix |
#         ecg 1500-2600, electrodes on | battery 3.95 V 82 % | wifi EACCESS-M1 -61 dBm | relay sent 862
I2C_NAMES = {"68": "MPU6050", "69": "MPU6050", "77": "BMP180", "1E": "HMC5883L", "0D": "QMC5883L", "2C": "QMC5883P"}


def _row(key, label, state, value, hint=""):
    return {"key": key, "label": label, "state": state, "value": value, "hint": hint}


def parse_check_line(text: str) -> list:
    """Turn one ``[check]`` line into rows of (state ok/warn/bad, value, what to do)."""
    parts = {}
    for chunk in text.replace("[check]", "", 1).split("|"):
        chunk = chunk.strip()
        if " " in chunk:
            name, rest = chunk.split(" ", 1)
            parts[name] = rest.strip()
    rows = []

    found = [a for a in parts.get("i2c", "").split() if re.fullmatch(r"[0-9A-F]{2}", a)]
    if "68" in found or "69" in found:
        rows.append(_row("bus", "Sensor bus", "ok", ", ".join(I2C_NAMES.get(a, "0x" + a) for a in found)))
    else:
        rows.append(_row("bus", "Sensor bus", "bad", "no MPU6050" if found else "nothing found",
                         "GY-87 SDA to GPIO 4, SCL to GPIO 5, VCC_IN to 3.3 V, GND to GND"))

    imu = parts.get("imu", "missing")
    m = re.match(r"([\d.]+) m/s2(?: at ([\d.]+) Hz)?(?:, loop paused (\d+) ms)?", imu)
    if not m:
        rows.append(_row("imu", "Motion", "bad", "no data", "Check the GY-87 wiring (sensor bus above)"))
    else:
        g, hz = float(m.group(1)), float(m.group(2) or 0)
        state = "ok" if 7.5 <= g <= 12.0 and (hz == 0 or hz >= 45) else "warn"
        hint = "" if state == "ok" else ("Fewer than 50 samples a second: the vest is overloaded" if hz and hz < 45
                                        else "Should read about 9.8 at rest; calibrate on the Vest readings page")
        rows.append(_row("imu", "Motion", state, f"{g:.1f} m/s² at {hz:.0f} Hz" if hz else f"{g:.1f} m/s²", hint))

    mag = parts.get("mag", "none")
    m = re.match(r"(\S+)\s+([\d.]+) uT", mag)
    if mag == "none":
        rows.append(_row("mag", "Heading", "warn", "no magnetometer", "Heading is unavailable; everything else works"))
    elif m:
        ut = float(m.group(2))
        rows.append(_row("mag", "Heading", "ok" if 15 <= ut <= 100 else "warn", f"{m.group(1)}, {ut:.0f} µT",
                         "" if 15 <= ut <= 100 else "Unusual field strength: keep the pod away from metal and magnets"))
    else:
        rows.append(_row("mag", "Heading", "warn", mag))

    gps = parts.get("gps", "")
    m = re.match(r"(\d+) chars, (\d+) sats, (fix|no fix)", gps)
    if m and m.group(3) == "fix":
        rows.append(_row("gps", "GPS", "ok", f"fix, {m.group(2)} satellites"))
    elif m:
        rows.append(_row("gps", "GPS", "warn", f"receiving, {m.group(2)} satellites",
                         "Waiting for a fix: outdoors, antenna facing the sky, up to a few minutes"))
    elif gps.startswith("bytes"):
        rows.append(_row("gps", "GPS", "warn", "data at an unknown speed", "Still trying 9600, 38400 and 115200 baud"))
    else:
        rows.append(_row("gps", "GPS", "bad", "no data",
                         "Check GPS VCC to 3.3 V, GND to GND and GPS TX to GPIO 16 (the vest also tries GPIO 17)"))

    ecg = parts.get("ecg", "")
    m = re.match(r"(\d+)-(\d+), electrodes (on|off)", ecg)
    if not m:
        rows.append(_row("ecg", "ECG", "bad", "no data", "AD8232 OUTPUT to GPIO 2, 3.3v through 10 Ω, LO+ to GPIO 11, LO- to GPIO 12"))
    else:
        lo, hi, on = int(m.group(1)), int(m.group(2)), m.group(3) == "on"
        if not on:
            rows.append(_row("ecg", "ECG", "warn", "electrodes off", "Put on three gel electrodes and plug the cable fully into the jack"))
        elif hi - lo < 20:
            rows.append(_row("ecg", "ECG", "bad", "flat signal", "AD8232 OUTPUT to GPIO 2 and its 3.3 V supply"))
        elif hi >= 4000 or lo <= 5:
            rows.append(_row("ecg", "ECG", "warn", f"signal {lo} to {hi}", "Signal hits the limit: press the electrodes on and keep still"))
        else:
            rows.append(_row("ecg", "ECG", "ok", f"signal {lo} to {hi}"))

    batt = parts.get("battery", "")
    m = re.match(r"([\d.]+) V ([\d.]+) %", batt)
    if m:
        pct = float(m.group(2))
        rows.append(_row("battery", "Battery", "ok" if pct >= 15 else "warn", f"{float(m.group(1)):.2f} V, {pct:.0f} %",
                         "" if pct >= 15 else "Charge soon"))
    else:
        rows.append(_row("battery", "Battery", "warn", "not measured",
                         "Wire the two 100 kΩ resistors from the switched battery line to GPIO 1"))

    wifi = parts.get("wifi", "not connected")
    m = re.match(r"(.+) (-\d+) dBm$", wifi)
    if m:
        rssi = int(m.group(2))
        rows.append(_row("wifi", "Wi-Fi", "ok" if rssi >= -80 else "warn", f"{m.group(1)}, {rssi} dBm",
                         "" if rssi >= -80 else "Weak signal: move closer to the access point"))
    else:
        rows.append(_row("wifi", "Wi-Fi", "bad", "not connected",
                         "Needs a 2.4 GHz network or hotspot; flash again with its name and password"))

    sent = parts.get("relay", "")
    m = re.match(r"sent (\d+)", sent)
    n = int(m.group(1)) if m else 0
    wifi_ok = rows[-1]["state"] != "bad"
    rows.append(_row("relay", "Relay", "ok" if n else ("warn" if wifi_ok else "bad"), f"{n} samples sent" if n else "nothing sent yet",
                     "" if n else ("Connected but nothing accepted: check the relay token and certificates" if wifi_ok else "Waiting for Wi-Fi")))
    return rows
