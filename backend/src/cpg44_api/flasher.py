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
        """Return only ports reported by the OS; an empty list means no USB pass-through."""
        ports = []
        try:
            import serial.tools.list_ports
            for port in serial.tools.list_ports.comports():
                ports.append({
                    "device": port.device,
                    "name": port.name,
                    "description": port.description,
                    "hwid": port.hwid,
                    "is_usb": "USB" in (port.hwid or "").upper()
                    or "UART" in (port.description or "").upper(),
                })
        except Exception:
            return []
        return ports

    def toolchain_status(self) -> dict:
        arduino_cli = shutil.which("arduino-cli")
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
            "message": (
                "Ready to compile and flash the relay-only firmware."
                if arduino_cli and ports and relay["ready"]
                else "Attach the ESP32, install Arduino CLI, and configure the relay token and CA file."
            ),
        }

    def _port_exists(self, port: str) -> bool:
        return any(item["device"] == port for item in self.list_ports())

    def get_chip_info(self, port: str) -> dict:
        if not self._port_exists(port):
            return {"ok": False, "port": port, "error": "serial port is not attached to WSL"}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "esptool", "--port", port, "chip-id"],
                capture_output=True,
                text=True,
                timeout=12,
            )
        except Exception as exc:
            return {"ok": False, "port": port, "error": str(exc)}
        output = (result.stdout + "\n" + result.stderr).strip()
        return {
            "ok": result.returncode == 0,
            "port": port,
            "output": output,
            "error": None if result.returncode == 0 else output or "esptool failed",
        }

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
        arduino_cli = shutil.which("arduino-cli")
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
                "--fqbn", "esp32:esp32:esp32s3", str(self.firmware_dir),
            ]
            # Arduino CLI documents compile and upload as separate commands.
            # Keeping them separate produces an accurate failure stage and avoids
            # relying on board-core-specific upload-property flags.
            compile_command = [
                arduino_cli, "compile", "--fqbn", "esp32:esp32:esp32s3",
                str(self.firmware_dir),
            ]
            return_code = 0
            for stage, stage_command in (("compile", compile_command), ("upload", upload_command)):
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
