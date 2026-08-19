"""
soccer_analytics.sensors
========================

Wearable-data fusion layer (CPG44 Objectives 2 & 3). The vision pipeline supplies
external load (position, speed, distance, HSR, sprints); this package ingests the
custom wearable's internal load (HR, SpO2, IMU) and fuses the two streams into
per-player features that drive injury-risk prediction and the recommendation
engine.

Hardware plugs in behind a single interface — :class:`SensorSource`. Today a
:class:`SimulatedSensorSource` stands in; swap it for :class:`SerialSensorSource`
(ESP32 over USB) or :class:`UdpSensorSource` (Wi-Fi microcontroller) with no
change to the rest of the pipeline.

Data flow::

    vision metrics ─┐
                    ├─▶ FusionEngine ─▶ WorkloadTracker ─▶ InjuryRiskModel ─┐
    SensorSource ───┘        (per-player features, time-aligned)           │
                                                                           ▼
                                                       RecommendationEngine
"""

from .schema import SensorSample, VisionSample, FusedSample, WorkloadFeatures, InjuryRisk
from .source import SensorSource, SimulatedSensorSource, SerialSensorSource, UdpSensorSource
from .hub_bridge import HubSensorSource, snapshot_to_sample, sample_to_observation
from .sync import SensorVideoSync
from .fusion import FusionEngine
from .injury import InjuryRiskModel, HeuristicInjuryModel
from .recommend import RecommendationEngine

__all__ = [
    "SensorSample", "VisionSample", "FusedSample", "WorkloadFeatures", "InjuryRisk",
    "SensorSource", "SimulatedSensorSource", "SerialSensorSource", "UdpSensorSource",
    "HubSensorSource", "snapshot_to_sample", "sample_to_observation",
    "SensorVideoSync", "FusionEngine",
    "InjuryRiskModel", "HeuristicInjuryModel", "RecommendationEngine",
]


def endpoint_source(host: str = "0.0.0.0", port: int = 8000, token=None):
    """Convenience: an HTTP/WebSocket-fed SensorSource (lazy import of FastAPI)."""
    from .server import EndpointSensorSource
    return EndpointSensorSource(host=host, port=port, token=token)
