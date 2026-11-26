#!/usr/bin/env bash
# Hand a USB device from Windows to WSL: the vest's ESP32 board or a phone in
# USB webcam mode. Uses usbipd-win (https://github.com/dorssel/usbipd-win).
#
#   scripts/wsl_usb.sh list              what Windows sees, and what is attached
#   scripts/wsl_usb.sh esp32             attach the vest board  -> /dev/ttyACM0
#   scripts/wsl_usb.sh phone             attach a phone webcam  -> /dev/video0
#   scripts/wsl_usb.sh phone 1-7         attach a given bus id
#   scripts/wsl_usb.sh detach esp32      give it back to Windows (or: detach phone, detach 1-7)
#   scripts/wsl_usb.sh watch esp32       re-attach whenever the board comes back (Ctrl+C to stop)
#
# The first attach of a device needs it "shared", which Windows only allows as
# administrator: the script asks for that once (a Windows prompt appears).
set -euo pipefail

USBIPD=""
for c in "$(command -v usbipd.exe 2>/dev/null || true)" \
         "/mnt/c/Program Files/usbipd-win/usbipd.exe" "/mnt/c/Program Files (x86)/usbipd-win/usbipd.exe"; do
  # a plain "usbipd" on the Linux PATH is the USB/IP server daemon, not this tool
  if [[ -n "$c" && -x "$c" ]]; then USBIPD="$c"; break; fi
done
if [[ -z "$USBIPD" ]]; then
  echo "usbipd-win is not installed on Windows. In PowerShell: winget install usbipd" >&2
  exit 2
fi

# USB vendor ids: Espressif and the usual USB-serial bridges on ESP32 boards
ESP32_VIDS="303a|10c4|1a86|0403"
# phone makers (Google, Samsung, Xiaomi, OnePlus, Motorola, OPPO/Realme, vivo, Huawei, Sony, Nothing)
PHONE_VIDS="18d1|04e8|2717|2a70|22b8|22d9|2d95|12d1|0fce|2b0e"
PHONE_WORDS="webcam|camera|uvc|android|pixel|galaxy|phone"

devices() { "$USBIPD" list 2>/dev/null | tr -d '\r' | awk '/^Persisted:/{exit} /^[0-9]+-[0-9.]+[[:space:]]/'; }

find_busid() {   # $1 = esp32|phone
  local line
  if [[ "$1" == esp32 ]]; then
    line=$(devices | grep -iE "^[^ ]+[[:space:]]+($ESP32_VIDS):" | head -1 || true)
  else
    line=$(devices | grep -iE "^[^ ]+[[:space:]]+($PHONE_VIDS):" | head -1 || true)
    [[ -z "$line" ]] && line=$(devices | grep -iE "$PHONE_WORDS" | grep -viE "integrated camera|bluetooth" | head -1 || true)
  fi
  awk '{print $1}' <<<"$line"
}

state_of() { devices | awk -v b="$1" '$1==b' | grep -oE "Not shared|Shared|Attached" | head -1; }

share() {        # bind needs administrator rights on Windows
  local busid=$1
  [[ "$(state_of "$busid")" != "Not shared" ]] && return 0
  echo "Sharing $busid with WSL (Windows asks for administrator rights once)..."
  powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '$(wslpath -w "$USBIPD")' -ArgumentList 'bind --busid $busid' -Verb RunAs -Wait" >/dev/null
  [[ "$(state_of "$busid")" != "Not shared" ]] || { echo "Sharing was not allowed; run as administrator: usbipd bind --busid $busid" >&2; exit 1; }
}

wait_for() {     # $1 = glob, $2 = seconds
  for _ in $(seq 1 "$2"); do compgen -G "$1" >/dev/null && { ls $1; return 0; }; sleep 1; done
  return 1
}

attach() {       # $1 = esp32|phone, $2 = optional bus id
  local kind=$1 busid=${2:-}
  [[ -z "$busid" ]] && busid=$(find_busid "$kind")
  if [[ -z "$busid" ]]; then
    if [[ "$kind" == esp32 ]]; then
      echo "No ESP32 board on Windows. Plug the vest in with a data cable (the port marked USB)." >&2
    else
      echo "No phone webcam on Windows. On the phone: plug in, tap 'Charging this device via USB', choose Webcam." >&2
      echo "Then run '$0 list' and pass its bus id: $0 phone <busid>" >&2
    fi
    exit 1
  fi
  if [[ "$(state_of "$busid")" == "Attached" ]]; then
    echo "$busid is already attached to WSL."
  else
    share "$busid"
    "$USBIPD" attach --wsl --busid "$busid" 2>&1 | tr -d '\r' | grep -v "^usbipd: info" || true
  fi
  if [[ "$kind" == esp32 ]]; then
    wait_for "/dev/ttyACM*" 10 || wait_for "/dev/ttyUSB*" 5 || { echo "Attached, but no serial port appeared yet." >&2; exit 1; }
  else
    # the webcam driver ships with the WSL kernel as a module
    lsmod | grep -q '^uvcvideo' || sudo modprobe uvcvideo
    if wait_for "/dev/video*" 10; then
      echo "In the app: Cameras > Other sources > USB camera, source 0."
    else
      echo "Attached, but no /dev/video device: is the phone set to Webcam (not File transfer)?" >&2; exit 1
    fi
  fi
}

case "${1:-list}" in
  list)
    "$USBIPD" list | tr -d '\r'
    echo; echo "In WSL:"; ls /dev/ttyACM* /dev/ttyUSB* /dev/video* 2>/dev/null || echo "  no serial ports or cameras"
    ;;
  esp32|phone) attach "$1" "${2:-}" ;;
  detach)
    target=${2:?say which: esp32, phone or a bus id}
    [[ "$target" == esp32 || "$target" == phone ]] && target=$(find_busid "$target")
    [[ -n "$target" ]] || { echo "Nothing to detach." >&2; exit 1; }
    "$USBIPD" detach --busid "$target" && echo "$target is back on Windows."
    ;;
  watch)
    kind=${2:-esp32}
    echo "Re-attaching the $kind whenever it is plugged in. Ctrl+C to stop."
    while true; do
      busid=$(find_busid "$kind")
      if [[ -n "$busid" && "$(state_of "$busid")" != "Attached" ]]; then attach "$kind" "$busid" || true; fi
      sleep 3
    done
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
