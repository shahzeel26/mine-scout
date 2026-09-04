from pathlib import Path
from datetime import datetime
import csv

from fastapi import FastAPI
from pydantic import BaseModel

DATA_FILE = Path("data/live_sensor_readings.csv")
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "timestamp",
    "location",
    "temperature_c",
    "distance_cm",
    "tilt_deg",
    "connection",
]

app = FastAPI(title="MineScout Rover Telemetry API")

class Telemetry(BaseModel):
    timestamp: str | None = None
    location: str = "Unknown"
    temperature_c: float
    distance_cm: float | None = None
    tilt_deg: float | None = None
    connection: str = "CONNECTED"

@app.get("/health")
def health():
    return {"status": "ok", "service": "MineScout Telemetry API"}

@app.get("/latest")
def latest():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        return {"data": None}

    with DATA_FILE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return {"data": rows[-1] if rows else None}

@app.post("/telemetry")
def telemetry(item: Telemetry):
    row = item.model_dump()

    if not row["timestamp"]:
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    write_header = not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0

    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in FIELDNAMES})

    return {
        "saved": True,
        "location": row["location"],
        "timestamp": row["timestamp"],
    }
