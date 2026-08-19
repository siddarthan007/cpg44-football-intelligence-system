"""Non-contact injury-risk prediction (CPG44 Objective 3).

Two interchangeable models sharing one interface (``predict(features) -> InjuryRisk``):

- :class:`HeuristicInjuryModel` — rule-based on established sports-science markers
  (ACWR sweet-spot 0.8-1.3 after Gabbett, cardiac drift, SpO2, HSR/load). Works
  with ZERO training data, so the pipeline is useful on day one.
- :class:`InjuryRiskModel` — gradient-boosted / random-forest model (the report's
  XGBoost/RandomForest approach) trained on historical
  ``WorkloadFeatures`` → injury-label data. Until real labels exist it can be
  bootstrapped from heuristic weak-labels (clearly flagged).

Feature order is fixed so a saved model and live features always align.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .schema import InjuryRisk, WorkloadFeatures

FEATURE_ORDER = ["total_distance", "hsr_distance", "sprint_count", "accel_efforts",
                 "decel_efforts", "player_load", "metabolic_power_avg",
                 "high_metabolic_distance", "energy_kcal", "top_speed",
                 "avg_hr", "hr_drift", "min_spo2", "acwr"]


def features_to_array(f: WorkloadFeatures) -> np.ndarray:
    v = f.to_vector()
    # NaNs (missing wearable) → neutral values so the model still runs vision-only
    defaults = {"avg_hr": 150.0, "hr_drift": 0.0, "min_spo2": 97.0, "acwr": 1.0}
    return np.array([defaults.get(k, 0.0) if (v[k] is None or np.isnan(v[k])) else v[k]
                     for k in FEATURE_ORDER], dtype=float)


def _level(risk: float) -> str:
    return "high" if risk >= 0.66 else "moderate" if risk >= 0.33 else "low"


class HeuristicInjuryModel:
    """Transparent rule-based baseline. No training required."""

    def predict(self, f: WorkloadFeatures) -> InjuryRisk:
        factors = {}

        # ACWR: risk rises sharply above 1.5, and also when undertrained (<0.8)
        acwr = f.acwr if not np.isnan(f.acwr) else 1.0
        if acwr >= 1.3:
            factors["acwr_high"] = float(np.clip((acwr - 1.3) / 0.7, 0, 1))
        elif acwr < 0.8:
            factors["acwr_low"] = float(np.clip((0.8 - acwr) / 0.8, 0, 1))

        # cardiac drift (late-vs-early HR) → accumulating fatigue
        if not np.isnan(f.hr_drift) and f.hr_drift > 8:
            factors["cardiac_drift"] = float(np.clip((f.hr_drift - 8) / 22, 0, 1))

        # desaturation
        if not np.isnan(f.min_spo2) and f.min_spo2 < 94:
            factors["low_spo2"] = float(np.clip((94 - f.min_spo2) / 6, 0, 1))

        # high external load (HSR + sprints) — relative to typical match maxima
        hsr_r = np.clip(f.hsr_distance / 1200.0, 0, 1)          # ~1.2 km HSR = heavy
        sprint_r = np.clip(f.sprint_count / 35.0, 0, 1)
        if hsr_r > 0.6 or sprint_r > 0.6:
            factors["high_intensity_load"] = float(max(hsr_r, sprint_r))

        # accel/decel load — eccentric decelerations are a key soft-tissue risk
        ad = f.accel_efforts + f.decel_efforts
        ad_r = np.clip(ad / 120.0, 0, 1)
        if ad_r > 0.6:
            factors["accel_decel_load"] = float(ad_r)

        # metabolic load (di Prampero) — high sustained energy expenditure
        met_r = np.clip(f.high_metabolic_distance / 1500.0, 0, 1)
        if met_r > 0.6:
            factors["metabolic_load"] = float(met_r)

        # weighted aggregation
        weights = {"acwr_high": 0.3, "acwr_low": 0.12, "cardiac_drift": 0.16,
                   "low_spo2": 0.12, "high_intensity_load": 0.12,
                   "accel_decel_load": 0.1, "metabolic_load": 0.08}
        risk = float(np.clip(sum(factors.get(k, 0) * w for k, w in weights.items())
                             / max(sum(weights[k] for k in factors), 1e-9)
                             * min(1.0, sum(factors.values())), 0, 1)) if factors else 0.05
        return InjuryRisk(f.player_id, round(risk, 3), _level(risk),
                          {k: round(v, 3) for k, v in factors.items()})


class InjuryRiskModel:
    """Trainable ML model (XGBoost if available, else RandomForest)."""

    def __init__(self):
        self.model = None
        self._backend = None

    def _new_model(self):
        try:
            from xgboost import XGBClassifier
            self._backend = "xgboost"
            return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, eval_metric="logloss")
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            self._backend = "random_forest"
            return RandomForestClassifier(n_estimators=400, max_depth=8,
                                          class_weight="balanced", random_state=0)

    def fit(self, features: List[WorkloadFeatures], labels: List[int]):
        X = np.vstack([features_to_array(f) for f in features])
        y = np.asarray(labels, dtype=int)
        self.model = self._new_model()
        self.model.fit(X, y)
        return self

    def predict(self, f: WorkloadFeatures) -> InjuryRisk:
        if self.model is None:
            raise RuntimeError("InjuryRiskModel not trained; fit() or use HeuristicInjuryModel")
        x = features_to_array(f).reshape(1, -1)
        risk = float(self.model.predict_proba(x)[0, 1])
        factors = self._importances(x)
        return InjuryRisk(f.player_id, round(risk, 3), _level(risk), factors)

    def _importances(self, x) -> dict:
        try:
            imp = getattr(self.model, "feature_importances_", None)
            if imp is None:
                return {}
            order = np.argsort(imp)[::-1][:3]
            return {FEATURE_ORDER[i]: round(float(imp[i]), 3) for i in order}
        except Exception:
            return {}

    def save(self, path: str):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"backend": self._backend, "model": self.model}, fh)

    @classmethod
    def load(cls, path: str) -> "InjuryRiskModel":
        import pickle
        obj = cls()
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj.model, obj._backend = d["model"], d["backend"]
        return obj


def bootstrap_training_set(n: int = 4000, seed: int = 0
                           ) -> Tuple[List[WorkloadFeatures], List[int]]:
    """Synthetic feature set weak-labelled by the heuristic — lets you train and
    validate the ML pipeline BEFORE real injury data exists. Replace with logged
    match+wearable+injury records for production use."""
    rng = np.random.default_rng(seed)
    heur = HeuristicInjuryModel()
    feats, labels = [], []
    for _ in range(n):
        f = WorkloadFeatures(
            player_id=0,
            total_distance=float(rng.uniform(3000, 13000)),
            hsr_distance=float(rng.uniform(100, 1600)),
            sprint_count=int(rng.integers(0, 45)),
            accel_efforts=int(rng.integers(0, 90)),
            decel_efforts=int(rng.integers(0, 90)),
            player_load=float(rng.uniform(200, 1000)),
            metabolic_power_avg=float(rng.uniform(6, 16)),
            high_metabolic_distance=float(rng.uniform(100, 2000)),
            energy_kcal=float(rng.uniform(400, 1400)),
            top_speed=float(rng.uniform(6, 10)),
            avg_hr=float(rng.uniform(120, 185)),
            hr_drift=float(rng.uniform(-5, 35)),
            min_spo2=float(rng.uniform(88, 99)),
            acwr=float(rng.uniform(0.5, 2.2)),
        )
        r = heur.predict(f).risk
        # probabilistic label around the heuristic risk (adds realistic noise)
        labels.append(int(rng.uniform() < r))
        feats.append(f)
    return feats, labels
