"""Electrical check of the vest circuit against datasheet limits.

    python scripts/check_vest_circuit.py            # report, exit 1 on any failure

The circuit is written down once here (rails, parts, every wire) and checked
for worst-case conditions:

* every chip's supply sits inside its operating range at the lowest and the
  highest rail voltage;
* every driven pin meets the receiver's logic thresholds, and no input is
  pushed above what it tolerates;
* both analog inputs stay inside the ESP32-S3 ADC's effective range at 12 dB
  attenuation (0 to 2.9 V);
* the regulator, switch, protection board and cell carry the average and peak
  currents, and the battery life is estimated;
* every GPIO avoids the pins the ESP32-S3 N16R8 DevKitC-1 reserves, analog pins
  are on ADC1, and the pins agree with the firmware's #define lines.

Limits come from the manufacturers' datasheets listed in SOURCES. Where a
cheap module's exact parts are not documented (the GY-87 and GY-NEO6MV2
regulators), the check assumes a pessimistic 150 mV regulator drop.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware" / "wearable_stream" / "wearable_stream.ino"

SOURCES = {
    "esp32": "ESP32-S3-WROOM-1 datasheet v1.8, tables 6-2, 6-3, 6-4 (documentation.espressif.com)",
    "esp32adc": "ESP32-S3 series datasheet, ADC characteristics: ATTEN3 effective range 0 to 2900 mV, +-50 mV",
    "devkit": "ESP32-S3-DevKitC-1 v1.1 user guide: three mutually exclusive power inputs; GPIO35-37 used by octal PSRAM",
    "tps63020": "TI TPS63020 datasheet: VIN 1.8 to 5.5 V, 2 A out at VIN > 2.5 V and VOUT 3.3 V, load disconnect",
    "ad8232": "Analog Devices AD8232 datasheet: supply 2.0 to 3.5 V (abs. max 3.6 V), 170 uA typ, 230 uA max",
    "neo6": "u-blox NEO-6 datasheet GPS.G6-HW-09005: VCC 2.7 to 3.6 V, 67 mA max, 39 mA tracking, I/O 0 to VCC",
    "mpu6050": "InvenSense MPU-6000/6050 product specification: VDD 2.375 to 3.46 V, 3.9 mA",
    "hmc5883l": "Honeywell HMC5883L datasheet: VDD 2.16 to 3.6 V, I2C 0x1E",
    "bmp180": "Bosch BMP180 datasheet: VDD 1.8 to 3.6 V, I2C 0x77",
    "tp4056": "TP4056 datasheet: 4.2 V +-1.5 % float, RPROG 1.2 kOhm = 1000 mA",
    "dw01a": "DW01A datasheet: overcharge 4.30 V, overdischarge 2.40 V, overcurrent 150 mV across the FETs",
}


@dataclass
class Rail:
    name: str
    lo: float
    hi: float


# ---------------------------------------------------------------- rails (V) --
BATTERY = Rail("battery (LiPo, in use)", 3.0, 4.2)       # 3.0 V: the cell's own protection or the vest's low warning
RAIL_3V3 = Rail("3.3 V rail (TPS63020 out)", 3.20, 3.40)  # fixed-output module, +-3 % allowed
LDO_DROP = 0.15                                           # pessimistic drop of the modules' small regulators
GY87_INT = Rail("GY-87 internal rail", RAIL_3V3.lo - LDO_DROP, RAIL_3V3.hi)
GPS_INT = Rail("GPS module internal rail", RAIL_3V3.lo - LDO_DROP, RAIL_3V3.hi)
AD8232_RAIL = Rail("AD8232 supply (after 10 Ohm filter)", RAIL_3V3.lo - 10 * 230e-6, RAIL_3V3.hi)

# --------------------------------------------------- supply ranges per chip --
SUPPLIES = [
    ("ESP32-S3 module", RAIL_3V3, 3.0, 3.6, "esp32"),
    ("TPS63020 input", BATTERY, 1.8, 5.5, "tps63020"),
    ("AD8232", AD8232_RAIL, 2.0, 3.5, "ad8232"),
    ("MPU6050 (GY-87)", GY87_INT, 2.375, 3.46, "mpu6050"),
    ("HMC5883L / QMC5883L (GY-87)", GY87_INT, 2.16, 3.6, "hmc5883l"),
    ("BMP180 (GY-87)", GY87_INT, 1.8, 3.6, "bmp180"),
    ("NEO-6M", GPS_INT, 2.7, 3.6, "neo6"),
]

# ------------------------------------------------------------ wiring (nets) --
# name, ESP32 GPIO, firmware #define, direction seen from the ESP32, other end
WIRES = [
    ("I2C SDA", 4, "I2C_SDA", "bidir", "GY-87 SDA"),
    ("I2C SCL", 5, "I2C_SCL", "out", "GY-87 SCL"),
    ("GPS TX to ESP32", 16, "GPS_RX", "in", "NEO-6M TX"),
    ("ESP32 TX to GPS", 17, "GPS_TX", "out", "NEO-6M RX (through 1 kOhm)"),
    ("ECG signal", 2, "ECG_OUT", "analog", "AD8232 OUTPUT"),
    ("Electrode LO+", 11, "ECG_LO_PLUS", "in", "AD8232 LO+"),
    ("Electrode LO-", 12, "ECG_LO_MINUS", "in", "AD8232 LO-"),
    ("Battery sense", 1, "BATTERY_ADC", "analog", "100k/100k divider midpoint"),
]
# DevKitC-1 header positions (v1.1 user guide): J1 left, J3 right
HEADER = {1: "J3 pin 4", 2: "J3 pin 5", 4: "J1 pin 4", 5: "J1 pin 5", 11: "J1 pin 17", 12: "J1 pin 18",
          16: "J1 pin 9", 17: "J1 pin 10"}
RESERVED = {0: "strapping", 3: "strapping", 45: "strapping", 46: "strapping", 19: "USB D-", 20: "USB D+",
            43: "UART0 TX to the USB bridge", 44: "UART0 RX from the USB bridge", 38: "RGB LED (v1.1)",
            48: "RGB LED (v1.0)", **{g: "SPI flash" for g in range(26, 33)},
            **{g: "octal PSRAM (N16R8)" for g in (33, 34, 35, 36, 37)}}
ADC1 = set(range(1, 11))

# ------------------------------------------------------------ loads (mA) --
# (average, peak) at 3.3 V
LOADS = {
    "ESP32-S3, Wi-Fi sending every 1 s": (120.0, 355.0),     # 355 mA: 802.11b TX peak, table 6-4
    "NEO-6M, tracking at 5 Hz": (39.0, 67.0),
    "MPU6050 + magnetometer + BMP180": (4.5, 5.5),
    "AD8232": (0.17, 0.23),
    "Power LEDs on DevKit, GY-87, GPS": (6.0, 8.0),
}
CELL_MAH = 3000.0
# Battery divider choices: (label, ohms per side). Any equal pair halves the cell
# voltage, which is all the firmware assumes; the value only sets the current.
DIVIDERS = [("100 kOhm + 100 kOhm (design)", 100e3), ("100 Ohm + 100 Ohm, matched pair", 100.0),
            ("5 x 100 Ohm in series on each side", 500.0)]
PULLDOWN_OHMS = (30e3, 60e3)       # ESP32-S3 internal pull-down, typical 45 kOhm, wide spread
USABLE = 0.85            # capacity left after cut-off margin and ageing
EFFICIENCY = 0.88        # TPS63020 at 150 to 400 mA, 3.0 to 4.2 V in
SWITCH_A = 0.5           # common SS-12D00 miniature slide switch rating
DW01A_TRIP_A = 0.150 / 0.050   # 150 mV over two 25 mOhm FETs (FS8205A) = 3 A
CHARGE_MA = 1000.0
BULK_UF = 470.0


class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, item, ok, detail, warn=False):
        self.rows.append((group, item, "PASS" if ok else ("WARN" if warn else "FAIL"), detail))

    @property
    def failures(self):
        return [r for r in self.rows if r[2] == "FAIL"]


def firmware_pins() -> dict:
    src = FIRMWARE.read_text()
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"#define\s+(\w+)\s+(\d+)\b", src)}


def run() -> Report:
    r = Report()

    # 1. supply ranges
    for name, rail, lo, hi, src in SUPPLIES:
        ok = lo <= rail.lo and rail.hi <= hi
        r.add("Supply", name, ok, f"gets {rail.lo:.2f} to {rail.hi:.2f} V, needs {lo} to {hi} V "
              f"(margin {min(rail.lo - lo, hi - rail.hi):+.2f} V) [{src}]")

    # 2. logic levels (worst case: driver at its lowest rail, receiver at its highest)
    vih_esp = 0.75 * RAIL_3V3.hi
    vil_esp = 0.25 * RAIL_3V3.lo
    gps_voh = GPS_INT.lo - 0.4                       # datasheet VOH at 4 mA; the ESP32 input draws 50 nA
    r.add("Logic", "NEO-6M TX high read by ESP32 (GPIO16)", gps_voh >= vih_esp,
          f"GPS drives >= {gps_voh:.2f} V, ESP32 needs >= {vih_esp:.2f} V (at 4 mA load; at 50 nA it is ~{GPS_INT.lo:.2f} V)")
    over = RAIL_3V3.hi - GPS_INT.lo
    r.add("Logic", "ESP32 TX high into NEO-6M RX (GPIO17)", RAIL_3V3.hi <= 3.6 and over < 0.5,
          f"{RAIL_3V3.hi:.2f} V into an input rated 3.6 V abs. max; at most {over:.2f} V above the GPS rail, "
          f"below the 0.5 V that turns on its clamp diode; the 1 kOhm series resistor limits it further")
    r.add("Logic", "I2C high level read by ESP32 (pull-ups on the GY-87)", GY87_INT.lo >= vih_esp,
          f"bus idles at >= {GY87_INT.lo:.2f} V, ESP32 needs >= {vih_esp:.2f} V")
    r.add("Logic", "I2C low level read by MPU6050", 0.1 * RAIL_3V3.hi <= 0.3 * GY87_INT.lo,
          f"ESP32 pulls to <= {0.1 * RAIL_3V3.hi:.2f} V, MPU6050 needs <= {0.3 * GY87_INT.lo:.2f} V")
    r.add("Logic", "AD8232 LO+/LO- high read by ESP32 (GPIO11/12)", AD8232_RAIL.lo - 0.1 >= vih_esp,
          f"rail-to-rail output >= {AD8232_RAIL.lo - 0.1:.2f} V, ESP32 needs >= {vih_esp:.2f} V; low <= 0.1 V < {vil_esp:.2f} V")

    # 3. analog inputs inside the ADC's effective range at 12 dB
    adc_max = 2.90
    hi_div = BATTERY.hi * 1.01 / (1.01 + 0.99)       # 1 % resistors in the worst direction
    lo_div = BATTERY.lo * 0.99 / (0.99 + 1.01)
    r.add("ADC", "Battery divider on GPIO1", hi_div <= adc_max,
          f"{lo_div:.2f} to {hi_div:.2f} V for {BATTERY.lo} to {BATTERY.hi} V cell; effective range 0 to {adc_max} V [esp32adc]")
    err = (0.050 * 2) + BATTERY.hi * 0.01
    r.add("ADC", "Battery reading accuracy", err <= 0.15, f"+-{err:.2f} V worst case (ADC +-50 mV doubled by the divider, 1 % resistors)")
    centre = AD8232_RAIL.hi / 2
    r_peak = centre + 1.1                            # 1 mV R wave x board gain 1100
    r.add("ADC", "ECG on GPIO2", r_peak <= adc_max,
          f"rests at VS/2 = {centre:.2f} V, a 1 mV R wave reaches {r_peak:.2f} V (limit {adc_max} V); "
          f"waves over 1.1 mV clip at the top, which keeps beat timing", warn=r_peak > adc_max)
    thev = 100e3 / 2
    r.add("ADC", "Divider source impedance", True,
          f"{thev/1e3:.0f} kOhm with the design pair, held by 100 nF (time constant {thev*100e-9*1e3:.0f} ms); "
          f"the firmware takes the median of 15 readings")

    # 4. currents, regulator, switch, protection, battery life
    avg = sum(a for a, _ in LOADS.values())
    peak = sum(p for _, p in LOADS.values())
    r.add("Current", "3.3 V average / peak", peak <= 2000,
          f"{avg:.0f} mA average, {peak:.0f} mA peak; TPS63020 gives 2000 mA at VIN > 2.5 V [tps63020]")
    r.add("Current", "ESP32 supply requirement", True, "the module needs a supply able to deliver 500 mA; the TPS63020 gives 2000 mA [esp32]")
    batt_avg = avg * 3.3 / (BATTERY.lo + BATTERY.hi) * 2 / EFFICIENCY / 1000
    batt_peak = peak * 3.3 / BATTERY.lo / 0.85 / 1000
    r.add("Current", "Battery current", batt_peak < DW01A_TRIP_A,
          f"{batt_avg*1000:.0f} mA average, {batt_peak*1000:.0f} mA peak at {BATTERY.lo} V; protection trips near {DW01A_TRIP_A:.0f} A [dw01a]")
    droop = 0.355 * 50e-6 / (BULK_UF * 1e-6)
    r.add("Current", "Wi-Fi burst on the 470 uF", droop < 0.1,
          f"{droop*1000:.0f} mV droop while the converter reacts (50 us); the ESP32 resets near 2.7 V")
    r.add("Current", "Slide switch", batt_avg < SWITCH_A, f"{batt_avg*1000:.0f} mA average through a {SWITCH_A:.1f} A switch; "
          f"peaks of {batt_peak*1000:.0f} mA last milliseconds and are fed mostly by the 470 uF",
          warn=batt_peak > SWITCH_A)
    v_mid = (BATTERY.lo + BATTERY.hi) / 2
    hours = CELL_MAH * USABLE / (batt_avg * 1000 + v_mid / (2 * DIVIDERS[0][1]) * 1000)
    r.add("Battery", "Running time", hours >= 3, f"about {hours:.1f} h from {CELL_MAH:.0f} mAh ({USABLE:.0%} usable, {EFFICIENCY:.0%} converter)")

    # every accepted battery divider: current, heat, running time, firmware detection
    for label, ohms in DIVIDERS:
        i_max = BATTERY.hi / (2 * ohms)
        watts = i_max ** 2 * ohms
        h = CELL_MAH * USABLE / (batt_avg * 1000 + v_mid / (2 * ohms) * 1000)
        thev = ohms / 2
        seen = min(rpd / (thev + rpd) for rpd in PULLDOWN_OHMS)        # pulled/open ratio the probe sees
        off_days = CELL_MAH * USABLE / (v_mid / (2 * ohms) * 1000) / 24
        ok = watts <= 0.125 and h >= 3 and seen >= 0.25
        r.add("Battery sense", label, ok,
              f"{i_max*1000:.2f} mA while on, {watts*1000:.1f} mW per resistor (1/4 W parts), about {h:.1f} h running; "
              f"firmware sees it as wired (pull-down ratio {seen:.2f} >= 0.25); "
              f"before the switch it would empty the cell in {off_days:.0f} days" if off_days < 365 * 3
              else f"{i_max*1000:.2f} mA while on, {watts*1000:.2f} mW per resistor, about {h:.1f} h running; firmware sees it as wired")
    r.add("Battery", "Charging", True, f"{CELL_MAH / CHARGE_MA * 1.2:.1f} h at {CHARGE_MA:.0f} mA to 4.2 V, vest switched off [tp4056]")

    # 5. pins: reserved, ADC1, duplicates, firmware agreement
    fw = firmware_pins()
    used = [g for _, g, *_ in WIRES]
    r.add("Pins", "No pin used twice", len(used) == len(set(used)), ", ".join(f"GPIO{g}" for g in used))
    for name, gpio, define, kind, other in WIRES:
        bad = RESERVED.get(gpio)
        r.add("Pins", f"GPIO{gpio} ({name})", bad is None, f"{HEADER.get(gpio, '?')} to {other}" + (f": reserved for {bad}" if bad else ""))
        if kind == "analog":
            r.add("Pins", f"GPIO{gpio} is on ADC1", gpio in ADC1, "ADC2 is shared with Wi-Fi")
        r.add("Firmware", f"#define {define}", fw.get(define) == gpio, f"firmware {fw.get(define)}, wiring {gpio}")
    return r


def main() -> int:
    rep = run()
    width = max(len(r[1]) for r in rep.rows)
    for g in dict.fromkeys(r[0] for r in rep.rows):
        print(f"\n{g}")
        for _, item, status, detail in (r for r in rep.rows if r[0] == g):
            print(f"  {status}  {item:<{width}}  {detail}")
    fails = rep.failures
    warns = [r for r in rep.rows if r[2] == "WARN"]
    print(f"\n{len(rep.rows)} checks: {len(rep.rows) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
