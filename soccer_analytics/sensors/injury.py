"""Load-review baseline and outcome-labelled model support.

Two interchangeable models sharing one interface (``predict(features) -> InjuryRisk``):

- :class:`HeuristicInjuryModel` — retained name for compatibility; it produces a
  transparent, non-medical load-review score from workload markers.
- :class:`InjuryRiskModel` — gradient-boosted / random-forest model (the report's
  XGBoost/RandomForest approach) trained on historical
  ``WorkloadFeatures`` → independently collected outcome labels.

Feature order is fixed so a saved model and live features always align.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .schema import InjuryRisk, WorkloadFeatures

FEATURE_ORDER = ["total_distance", "hsr_distance", "sprint_count", "accel_efforts",
                 "decel_efforts", "player_load", "metabolic_power_avg",
                 "high_metabolic_distance", "energy_kcal", "top_speed",
                 "avg_hr", "hr_drift", "min_spo2", "acwr"]


def features_to_array(f: WorkloadFeatures) -> np.ndarray:
    v = f.to_vector()
    # Model-only neutral imputation. Missing readings remain null in the API/UI.
    defaults = {"avg_hr": 150.0, "hr_drift": 0.0, "min_spo2": 97.0, "acwr": 1.0}
    return np.array([defaults.get(k, 0.0) if (v[k] is None or np.isnan(v[k])) else v[k]
                     for k in FEATURE_ORDER], dtype=float)


def _level(risk: float) -> str:
    return "high" if risk >= 0.66 else "moderate" if risk >= 0.33 else "low"


class HeuristicInjuryModel:
    """Compatibility name for a transparent, non-medical load indicator."""

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
            raise RuntimeError("Outcome model not trained; collect and label real sessions first")
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
