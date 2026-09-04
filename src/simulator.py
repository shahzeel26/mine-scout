from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def simulate_mission(n_points=18, seed=42):
    """
    Simulated mission used before the physical rover is connected.
    The first five points are normal calibration observations.
    """
    rng = np.random.default_rng(seed)
    start = datetime.now().replace(microsecond=0)
    rows = []

    for i in range(n_points):
        temp = float(rng.normal(30.0, 0.8))
        tilt = float(abs(rng.normal(3.2, 1.3)))
        distance = float(max(18.0, rng.normal(92.0, 17.0)))

        # Deliberate demo events after calibration.
        if i == 8:
            temp = 35.2
        if i == 12:
            temp = 47.8
        if i == 15:
            tilt = 11.5

        rows.append({
            "timestamp": (start + timedelta(seconds=i * 5)).isoformat(),
            "location": f"Point_{i+1:02d}",
            "temperature_c": round(temp, 1),
            "distance_cm": round(distance, 1),
            "tilt_deg": round(tilt, 1),
            "connection": "CONNECTED",
        })

    return pd.DataFrame(rows)
