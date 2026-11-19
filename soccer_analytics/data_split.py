"""Sequence-disjoint validation splits for converted SoccerNet training frames."""

import hashlib
from pathlib import Path
import re
import yaml


def validate_training_data(path):
    """Reject an explicit test-as-validation configuration before model training."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid dataset YAML: {error}") from error
    if not isinstance(data, dict) or not data.get("train") or not data.get("val"):
        raise ValueError("dataset YAML must declare train and val splits")
    validation = data["val"] if isinstance(data["val"], list) else [data["val"]]
    tests = data.get("test", [])
    tests = tests if isinstance(tests, list) else [tests]
    for entry in validation:
        if any(part.lower() in {"test", "test.txt"} for part in Path(str(entry)).parts) or entry in tests:
            raise ValueError("test data cannot be used for model selection; prepare a sequence-disjoint val split")
    return data


def split_sequences(files, validation_fraction=0.2, seed=44):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    groups = {}
    for value in files:
        path = Path(value)
        match = re.fullmatch(r"(SNMOT-\d+)_\d+", path.stem)
        if match is None:
            raise ValueError(f"cannot infer sequence from {path.name}; provide explicit match groups")
        groups.setdefault(match[1], []).append(path)
    if len(groups) < 2:
        raise ValueError("need at least two independent sequences for validation")
    ordered = sorted(groups, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    nval = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    val_groups = set(ordered[:nval])
    train = sorted(p for group, paths in groups.items() if group not in val_groups for p in paths)
    val = sorted(p for group, paths in groups.items() if group in val_groups for p in paths)
    return train, val, {"seed": seed, "train_sequences": sorted(set(groups) - val_groups),
                        "validation_sequences": sorted(val_groups),
                        "test_policy": "Official test directory is not read. Group by original match when available."}
