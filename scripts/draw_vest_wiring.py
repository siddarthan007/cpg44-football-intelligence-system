"""Draw the vest wiring diagram: docs/hardware/vest_wiring.svg

    python scripts/draw_vest_wiring.py

Every wire, pin label and part value comes from the same plan that
scripts/check_vest_circuit.py verifies, so the drawing and the checked circuit
cannot drift apart unnoticed (tests/test_vest_circuit.py runs both).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "hardware" / "vest_wiring.svg"
spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts" / "check_vest_circuit.py")
chk = importlib.util.module_from_spec(spec)
sys.modules["chk"] = chk
spec.loader.exec_module(chk)

C = {"v33": "#d1402c", "gnd": "#1b2320", "bat": "#e07b00", "i2c": "#2251a3", "uart": "#1f8a5b",
     "ecg": "#b0306a", "sense": "#7a5af0", "box": "#fbf9f4", "edge": "#3c4540", "muted": "#5f645e"}
W, H = 1500, 1010
svg: list[str] = []


def text(x, y, s, size=13, weight=400, fill="#1b2320", anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def box(x, y, w, h, title, sub=""):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{C["box"]}" stroke="{C["edge"]}" stroke-width="1.6"/>')
    text(x + w / 2, y + 22, title, 15, 700, anchor="middle")
    if sub:
        text(x + w / 2, y + 40, sub, 11.5, 400, C["muted"], "middle")


def pin(x, y, label, side="left"):
    svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#fff" stroke="{C["edge"]}" stroke-width="1.4"/>')
    if side == "left":
        text(x + 9, y + 4.5, label, 12)
    elif side == "right":
        text(x - 9, y + 4.5, label, 12, anchor="end")
    else:
        text(x, y + 20 if side == "top" else y - 10, label, 12, anchor="middle")


def wire(points, color, width=2.4, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{extra}/>')


def dot(x, y, color):
    svg.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>')


def resistor(x, y, vertical, label, color):
    if vertical:
        svg.append(f'<rect x="{x - 7}" y="{y}" width="14" height="36" rx="2" fill="#fff" stroke="{color}" stroke-width="2"/>')
        text(x + 13, y + 22, label, 12, 600)
    else:
        svg.append(f'<rect x="{x}" y="{y - 7}" width="36" height="14" rx="2" fill="#fff" stroke="{color}" stroke-width="2"/>')
        text(x + 18, y - 12, label, 11.5, 600, anchor="middle")


def cap(x, y1, y2, label, polar=False, top=None):
    """Vertical capacitor between y1 (top, +) and y2 (bottom)."""
    mid = (y1 + y2) / 2
    wire([(x, y1), (x, mid - 4)], top or C["v33"], 2)
    wire([(x, mid + 4), (x, y2)], C["gnd"], 2)
    svg.append(f'<line x1="{x - 11}" y1="{mid - 4}" x2="{x + 11}" y2="{mid - 4}" stroke="{C["edge"]}" stroke-width="2.4"/>')
    if polar:
        svg.append(f'<path d="M{x - 11},{mid + 7} Q{x},{mid + 1} {x + 11},{mid + 7}" fill="none" stroke="{C["edge"]}" stroke-width="2.4"/>')
        text(x - 15, mid - 7, "+", 12, 700, anchor="end")
    else:
        svg.append(f'<line x1="{x - 11}" y1="{mid + 4}" x2="{x + 11}" y2="{mid + 4}" stroke="{C["edge"]}" stroke-width="2.4"/>')
    text(x + 15, mid + 4, label, 11.5, 600)


def gnd_symbol(x, y):
    for i, w in enumerate((14, 9, 4)):
        svg.append(f'<line x1="{x - w}" y1="{y + i * 4}" x2="{x + w}" y2="{y + i * 4}" stroke="{C["gnd"]}" stroke-width="2"/>')


# ------------------------------------------------------------------ power --
R33, RGND = 262, 318                         # rail heights
box(40, 70, 170, 120, "LiPo cell", "3.7 V 3000 mAh")
pin(210, 110, "+", "right"); pin(210, 160, "-", "right")
text(125, 180, "red +, black -: check!", 11, 600, C["v33"], "middle")
box(290, 60, 210, 150, "TP4056 charger", "with protection (DW01A)")
pin(290, 110, "B+"); pin(290, 160, "B-"); pin(500, 110, "OUT+", "right"); pin(500, 160, "OUT-", "right")
text(395, 200, "USB in: 5 V charging only", 11.5, 400, C["muted"], "middle")
box(580, 70, 120, 80, "SPDT switch", "")
pin(580, 110, "COM"); pin(700, 110, "ON", "right")
text(640, 140, "third pin: unused", 11, 400, C["muted"], "middle")
box(790, 60, 200, 150, "TPS63020", "buck-boost, fixed 3.3 V")
pin(790, 110, "VIN"); pin(790, 160, "GND"); pin(990, 110, "VOUT", "right"); pin(990, 160, "GND", "right")
text(890, 200, "EN tied to VIN on the board", 11, 400, C["muted"], "middle")

wire([(210, 110), (290, 110)], C["v33"]); wire([(210, 160), (290, 160)], C["gnd"])
wire([(500, 110), (580, 110)], C["bat"]); wire([(700, 110), (790, 110)], C["bat"])
wire([(500, 160), (790, 160)], C["gnd"])
dot(745, 110, C["bat"])                                   # switched battery node for the divider
text(380, 32, "VBAT after the switch, to the battery sense divider", 11.5, 600, C["bat"])

# power plug between the converter and everything else
wire([(990, 110), (1060, 110)], C["v33"]); wire([(990, 160), (1060, 160)], C["gnd"])
svg.append(f'<rect x="1060" y="92" width="44" height="86" rx="5" fill="#fff" stroke="{C["edge"]}" stroke-width="1.6"/>')
text(1082, 116, "J", 12, 700, anchor="middle"); text(1082, 132, "S", 12, 700, anchor="middle"); text(1082, 148, "T", 12, 700, anchor="middle")
text(1115, 136, "Power plug: unplug it", 12, 700, C["v33"])
text(1115, 152, "before any USB cable", 12, 700, C["v33"])
wire([(1104, 110), (1440, 110), (1440, R33)], C["v33"])
wire([(1104, 160), (1470, 160), (1470, RGND)], C["gnd"])
wire([(360, R33), (1440, R33)], C["v33"], 3.2); text(370, R33 - 8, "3.3 V rail", 12, 700, C["v33"])
wire([(360, RGND), (1470, RGND)], C["gnd"], 3.2); text(660, RGND + 18, "GND rail (one common ground)", 12, 700, C["gnd"])
wire([(1440, R33), (1440, 930)], C["v33"], 3.2); wire([(1470, RGND), (1470, 930)], C["gnd"], 3.2)
dot(1440, R33, C["v33"]); dot(1470, RGND, C["gnd"])
cap(600, R33, RGND, "470 uF (stripe to GND)", polar=True)
cap(800, R33, RGND, "100 nF")
dot(600, R33, C["v33"]); dot(600, RGND, C["gnd"]); dot(800, R33, C["v33"]); dot(800, RGND, C["gnd"])

# ------------------------------------------------------------------ ESP32 --
ex, ey, ew, eh = 300, 380, 340, 560
box(ex, ey, ew, eh, "ESP32-S3-DevKitC-1", "N16R8 (WROOM-1)")
svg.append(f'<circle cx="390" cy="{ey}" r="4" fill="#fff" stroke="{C["edge"]}" stroke-width="1.4"/>')
svg.append(f'<circle cx="500" cy="{ey}" r="4" fill="#fff" stroke="{C["edge"]}" stroke-width="1.4"/>')
text(397, ey - 8, "3V3 (J1-1)", 12); text(507, ey - 8, "GND (J1-22)", 12)
wire([(390, ey), (390, R33)], C["v33"]); dot(390, R33, C["v33"])
wire([(500, ey), (500, RGND)], C["gnd"]); dot(500, RGND, C["gnd"])
rows = {4: 455, 5: 495, 16: 580, 17: 620, 2: 720, 11: 760, 12: 800}
labels = {4: "GPIO4 SDA (J1-4)", 5: "GPIO5 SCL (J1-5)", 16: "GPIO16 RX (J1-9)", 17: "GPIO17 TX (J1-10)",
          2: "GPIO2 ADC (J3-5)", 11: "GPIO11 (J1-17)", 12: "GPIO12 (J1-18)"}
for g, y in rows.items():
    pin(ex + ew, y, labels[g], "right")
pin(ex, 880, "GPIO1 ADC (J3-4)")
text(ex + ew / 2, 920, "USB: flash and serial monitor", 11.5, 400, C["muted"], "middle")

# battery divider (left)
wire([(745, 110), (745, 40), (20, 40), (20, 820)], C["bat"])
resistor(20, 820, True, "R1", C["bat"])
wire([(20, 856), (20, 880), (ex, 880)], C["sense"]); dot(20, 880, C["sense"])
resistor(20, 900, True, "R2", C["gnd"])
wire([(20, 880), (20, 900)], C["sense"]); wire([(20, 936), (20, 960)], C["gnd"]); gnd_symbol(20, 960)
wire([(150, 880), (150, 900)], C["sense"]); dot(150, 880, C["sense"])
cap(150, 900, 950, "100 nF", polar=False, top=C["sense"]); gnd_symbol(150, 950)
text(60, 990, "Battery sense: GPIO1 reads half the cell voltage (1.5 to 2.1 V)", 12, 600, C["sense"])
# divider choices, from the checked list in check_vest_circuit.py
text(40, 772, "R1 = R2, any equal pair:", 11.5, 700, C["sense"])
text(40, 788, "100 kOhm 1 % (14.4 h), or a matched", 11, 400, C["muted"])
text(40, 802, "100 Ohm pair (21 mA, 13.1 h)", 11, 400, C["muted"])

# ------------------------------------------------------------------ GY-87 --
gx, gw = 1010, 300
box(gx, 385, gw, 140, "GY-87 (10-DoF)", "MPU6050 0x68, mag 0x1E or 0x0D, BMP180 0x77")
pin(gx, 455, "SDA"); pin(gx, 495, "SCL")
pin(gx + gw, 455, "VCC_IN", "right"); pin(gx + gw, 495, "GND", "right")
text(gx + 150, 518, "3.3V, FSYNC, INTA, DRDY: not connected", 11, 400, C["muted"], "middle")
wire([(ex + ew, 455), (gx, 455)], C["i2c"]); wire([(ex + ew, 495), (gx, 495)], C["i2c"])
wire([(gx + gw, 455), (1440, 455)], C["v33"]); dot(1440, 455, C["v33"])
wire([(gx + gw, 495), (1470, 495)], C["gnd"]); dot(1470, 495, C["gnd"])

# ------------------------------------------------------------------ NEO-6M --
box(gx, 545, gw, 110, "NEO-6M GPS (GY-NEO6MV2)", "UART 9600 8N1; antenna face up")
pin(gx, 580, "TX"); pin(gx, 620, "RX")
pin(gx + gw, 580, "VCC", "right"); pin(gx + gw, 620, "GND", "right")
wire([(ex + ew, 580), (gx, 580)], C["uart"])
wire([(ex + ew, 620), (790, 620)], C["uart"]); resistor(790, 620, False, "1 kOhm", C["uart"]); wire([(826, 620), (gx, 620)], C["uart"])
wire([(gx + gw, 580), (1440, 580)], C["v33"]); dot(1440, 580, C["v33"])
wire([(gx + gw, 620), (1470, 620)], C["gnd"]); dot(1470, 620, C["gnd"])

# ------------------------------------------------------------------ AD8232 --
box(gx, 690, gw, 160, "AD8232 ECG", "3.3 V only, never 5 V")
pin(gx, 720, "OUTPUT"); pin(gx, 760, "LO+"); pin(gx, 800, "LO-")
pin(gx + gw, 720, "3.3v", "right"); pin(gx + gw, 760, "SDN", "right"); pin(gx + gw, 800, "GND", "right")
text(gx + 150, 840, "3.5 mm jack: RA, LA, RL electrodes", 11.5, 600, C["ecg"], "middle")
wire([(ex + ew, 720), (gx, 720)], C["ecg"]); wire([(ex + ew, 760), (gx, 760)], C["ecg"]); wire([(ex + ew, 800), (gx, 800)], C["ecg"])
wire([(gx + gw, 720), (1390, 720)], C["v33"]); resistor(1390, 720, False, "10 Ohm", C["v33"]); wire([(1426, 720), (1440, 720)], C["v33"])
dot(1440, 720, C["v33"])
wire([(gx + gw, 760), (1335, 760), (1335, 720)], C["v33"]); dot(1335, 720, C["v33"])
wire([(1362, 720), (1362, 735)], C["v33"]); dot(1362, 720, C["v33"])
cap(1362, 735, 795, "10 uF", polar=False)
wire([(1362, 795), (1362, 800)], C["gnd"])
wire([(gx + gw, 800), (1470, 800)], C["gnd"]); dot(1470, 800, C["gnd"]); dot(1362, 800, C["gnd"])

# ------------------------------------------------------------------ legend --
lx, ly = 1010, 885
items = [("v33", "3.3 V"), ("gnd", "ground"), ("bat", "battery"), ("i2c", "I2C"), ("uart", "GPS serial"), ("ecg", "ECG"), ("sense", "battery sense")]
for i, (k, name) in enumerate(items):
    x = lx + (i % 4) * 110
    y = ly + (i // 4) * 24
    wire([(x, y), (x + 26, y)], C[k], 3)
    text(x + 32, y + 4.5, name, 12)
text(lx, ly + 62, "A dot joins wires; wires that cross without a dot are not connected.", 11.5, 400, C["muted"])
text(lx, ly + 80, "Checked by scripts/check_vest_circuit.py against the datasheets.", 11.5, 400, C["muted"])

head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Archivo, Segoe UI, Arial, sans-serif" role="img" aria-label="CPG44 vest wiring diagram">'
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
OUT.write_text(head + "".join(svg) + "</svg>\n")
print(f"wrote {OUT.relative_to(ROOT)}")
