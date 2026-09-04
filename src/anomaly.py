import numpy as np
import pandas as pd

def _robust_stats(values, minimum_scale):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 0.0, minimum_scale

    centre = float(values.median())
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, minimum_scale)
    return centre, scale

def _score(z):
    return float(np.clip(abs(z) / 4.0 * 100.0, 0.0, 100.0))

def enrich_mission(df, baseline_points=5):
    """
    Multi-sensor anomaly engine.

    Temperature and tilt are evaluated against robust mission baselines.
    Distance is used as an operational proximity signal, not a safety judgement.
    """
    out = df.copy()
    if out.empty:
        return out

    n = min(len(out), max(3, min(int(baseline_points), len(out))))

    temp_centre, temp_scale = _robust_stats(
        out["temperature_c"].iloc[:n], minimum_scale=0.75
    )
    tilt_centre, tilt_scale = _robust_stats(
        out["tilt_deg"].iloc[:n], minimum_scale=1.0
    )

    out["temp_baseline_c"] = round(temp_centre, 2)
    out["temp_delta_c"] = (
        pd.to_numeric(out["temperature_c"]) - temp_centre
    ).round(2)
    out["temp_z"] = (
        (pd.to_numeric(out["temperature_c"]) - temp_centre) / temp_scale
    ).round(2)

    out["tilt_baseline_deg"] = round(tilt_centre, 2)
    out["tilt_delta_deg"] = (
        pd.to_numeric(out["tilt_deg"]) - tilt_centre
    ).round(2)
    out["tilt_z"] = (
        (pd.to_numeric(out["tilt_deg"]) - tilt_centre) / tilt_scale
    ).round(2)

    out["thermal_score"] = out["temp_z"].apply(_score).round(1)
    out["tilt_score"] = out["tilt_z"].apply(_score).round(1)

    # Proximity signal: closer objects increase the operational attention score.
    distance = pd.to_numeric(out["distance_cm"], errors="coerce").fillna(999)
    out["proximity_score"] = np.select(
        [distance < 25, distance < 45, distance < 70],
        [100.0, 70.0, 35.0],
        default=0.0,
    )

    # Weighted multi-sensor score.
    out["anomaly_score"] = (
        0.70 * out["thermal_score"]
        + 0.20 * out["tilt_score"]
        + 0.10 * out["proximity_score"]
    ).clip(0, 100).round(1)

    def label(score):
        if score >= 70:
            return "HIGH"
        if score >= 40:
            return "WATCH"
        return "NORMAL"

    out["status"] = out["anomaly_score"].apply(label)

    # Calibration observations are clearly marked and not treated as mission anomalies.
    out.loc[out.index[:n], "status"] = "BASELINE"
    out.loc[out.index[:n], "anomaly_score"] = 0.0

    return out
