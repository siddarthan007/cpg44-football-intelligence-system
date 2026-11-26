"""The vest circuit meets every datasheet limit and matches the firmware pins."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_vest_circuit", Path(__file__).resolve().parents[1] / "scripts" / "check_vest_circuit.py")
check = importlib.util.module_from_spec(spec)
sys.modules["check_vest_circuit"] = check
spec.loader.exec_module(check)


def test_no_electrical_or_pin_failures():
    rep = check.run()
    assert not rep.failures, rep.failures
    assert len(rep.rows) >= 40


def test_build_guide_pin_table_matches_the_wiring():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "hardware" / "BUILD_GUIDE.md").read_text()
    for _, gpio, _, _, _ in check.WIRES:
        assert f"GPIO {gpio} " in guide or f"GPIO{gpio} " in guide, gpio
