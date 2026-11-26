"""Automatic ESP32 USB pass-through from Windows into WSL via usbipd.

WSL 2 cannot see USB serial devices until the Windows-side ``usbipd`` tool
binds and attaches them. This module finds the ESP32 on the Windows host and
runs that sequence automatically, elevating through UAC only when required, so
the dashboard flasher page works without manual PowerShell steps.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Dict, List, Optional

# Espressif USB JTAG/serial plus the common UART bridge chips used on dev boards.
ESP32_NAME_TAGS = ("esp32", "jtag", "serial", "cp210", "ch340", "ch9102", "ftdi",
                   "303a:", "10c4:", "1a86:", "0403:")   # Espressif, Silicon Labs, WCH, FTDI USB vendor ids
BUSID_PATTERN = re.compile(r"^(\d+-\d+(?:\.\d+)*)\s")


def _usbipd_exe() -> Optional[str]:
    """The Windows usbipd-win tool. A plain ``usbipd`` on the Linux PATH is
    the USB/IP server daemon from linux-tools, which cannot list or attach
    Windows devices, so only an .exe counts."""
    path = shutil.which("usbipd.exe")
    if path:
        return path
    # WSL can execute Windows binaries directly; check the standard location.
    for candidate in (
        "/mnt/c/Program Files/usbipd-win/usbipd.exe",
        "/mnt/c/Program Files (x86)/usbipd-win/usbipd.exe",
    ):
        try:
            if Path_exists(candidate):
                return candidate
        except Exception:
            continue
    return None


def Path_exists(path: str) -> bool:
    import os
    return os.path.isfile(path)


def _run(cmd: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _run_elevated(cmd: List[str], timeout: int = 90) -> subprocess.CompletedProcess:
    """Run a Windows command elevated through UAC via Start-Process -Verb RunAs."""
    powershell = shutil.which("powershell.exe") or "powershell.exe"
    args = " ".join(f"'{part}'" for part in cmd)
    script = f"Start-Process {args.split()[0]} -Verb RunAs -Wait -ArgumentList {args}"
    return subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        capture_output=True, text=True, timeout=timeout,
    )


def list_windows_devices() -> List[Dict[str, str]]:
    """Parse `usbipd list` output into rows with busid, description, state."""
    exe = _usbipd_exe()
    if exe is None:
        return []
    try:
        result = _run([exe, "list"])
    except Exception:
        return []
    rows = []
    for line in (result.stdout or "").splitlines():
        match = BUSID_PATTERN.match(line.strip())
        if not match:
            continue
        busid = match.group(1)
        rest = line[match.end():].strip()
        state = "Unknown"
        for known in ("Attached", "Shared", "Not shared", "Forced"):
            if known.lower() in rest.lower():
                state = known
                break
        description = rest
        rows.append({"busid": busid, "description": description, "state": state})
    return rows


def find_esp32_busids() -> List[Dict[str, str]]:
    """Windows devices whose name looks like an ESP32 or serial bridge."""
    hits = []
    for row in list_windows_devices():
        text = row["description"].lower()
        if any(tag in text for tag in ESP32_NAME_TAGS):
            hits.append(row)
    return hits


def wsl_has_port() -> bool:
    """True when at least one real USB serial device exists inside WSL."""
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            device = port.device or ""
            hwid = (port.hwid or "").upper()
            if device.startswith("/dev/ttyS") and hwid in ("", "N/A"):
                continue
            if "USB" in hwid or device.startswith("/dev/ttyACM"):
                return True
    except Exception:
        pass
    return False


def ensure_esp32_usb_attached() -> dict:
    """Bind + attach the first ESP32-looking Windows device into WSL.

    Returns {"ok": True, "attached": bool, "busid": str} on success. An already
    attached device is reported with attached=False so callers can skip work."""
    exe = _usbipd_exe()
    if exe is None:
        return {
            "ok": False,
            "error": "usbipd-win is not installed on the Windows host. "
                     "Install it from https://github.com/dorssel/usbipd-win/releases",
        }
    if wsl_has_port():
        return {"ok": True, "attached": False, "reason": "a USB serial port is already present in WSL"}

    candidates = find_esp32_busids()
    if not candidates:
        return {
            "ok": False,
            "error": "no ESP32 or serial bridge found on the Windows side. "
                     "Connect the board and check Device Manager.",
        }

    errors = []
    for row in candidates:
        busid = row["busid"]
        lowered = row["description"].lower()
        if "attached" in lowered and "not shared" not in lowered:
            return {"ok": True, "attached": False, "busid": busid,
                    "reason": "already attached to WSL"}

        # bind needs elevation the first time per device
        bind = _run([exe, "bind", "--busid", busid])
        if bind.returncode != 0 and "administrator" in (bind.stderr or "").lower():
            elevated = _run_elevated([exe, "bind", "--busid", busid])
            if elevated.returncode != 0:
                errors.append(f"bind {busid}: elevation declined or failed")
                continue
        attach = _run([exe, "attach", "--wsl", "--busid", busid])
        if attach.returncode == 0:
            return {"ok": True, "attached": True, "busid": busid}
        message = (attach.stderr or attach.stdout or "").strip().splitlines()
        detail = message[-1] if message else "attach failed"
        if "busy" in detail.lower() or "used by windows" in detail.lower():
            detach = _run_elevated([exe, "detach", "--busid", busid])
            retry = _run([exe, "attach", "--wsl", "--busid", busid])
            if retry.returncode == 0:
                return {"ok": True, "attached": True, "busid": busid}
            force = _run_elevated([exe, "attach", "--wsl", "--busid", busid, "--force"])
            if force.returncode == 0:
                return {"ok": True, "attached": True, "busid": busid}
            errors.append(f"attach {busid}: {detail}")
            continue
        if "administrator" in detail.lower() or "access denied" in detail.lower():
            elevated = _run_elevated([exe, "attach", "--wsl", "--busid", busid])
            if elevated.returncode == 0:
                return {"ok": True, "attached": True, "busid": busid}
        errors.append(f"attach {busid}: {detail}")

    return {
        "ok": False,
        "error": "could not attach the ESP32: " + "; ".join(errors or ["unknown failure"]),
    }
