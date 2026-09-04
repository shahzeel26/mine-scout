from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil

EVENTS_FILE = Path("data/events.json")
EVENT_EVIDENCE_DIR = Path("evidence/events")


def _ensure_storage():
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVENT_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENTS_FILE.exists():
        EVENTS_FILE.write_text("[]", encoding="utf-8")


def load_events():
    _ensure_storage()
    try:
        return json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_events(events):
    _ensure_storage()
    EVENTS_FILE.write_text(
        json.dumps(events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _event_key(mission_name, row):
    return f"{mission_name}|{row['timestamp']}|{row['location']}"


def _reason(row):
    reasons = []

    if float(row.get("thermal_score", 0)) >= 40:
        reasons.append(
            f"thermal deviation {float(row.get('temp_delta_c', 0)):+.1f} °C"
        )

    if float(row.get("tilt_score", 0)) >= 40:
        reasons.append(
            f"tilt deviation {float(row.get('tilt_delta_deg', 0)):+.1f}°"
        )

    if float(row.get("proximity_score", 0)) >= 40:
        reasons.append(
            f"object proximity {float(row.get('distance_cm', 0)):.0f} cm"
        )

    return ", ".join(reasons) if reasons else "combined multi-sensor deviation"


def ensure_sensor_event(
    mission_name,
    row,
    source,
    demo_evidence_path=None,
):
    """
    Create one event for a WATCH/HIGH observation.
    Re-running the dashboard will not duplicate the same event.
    """
    if row["status"] not in {"WATCH", "HIGH"}:
        return None, False

    events = load_events()
    key = _event_key(mission_name, row)

    for event in events:
        if event.get("event_key") == key:
            return event, False

    event_id = f"EVT-{len(events) + 1:03d}"

    event = {
        "event_id": event_id,
        "event_key": key,
        "mission_name": mission_name,
        "source": source,
        "point": str(row["location"]),
        "timestamp": str(row["timestamp"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classification": str(row["status"]),
        "anomaly_score": float(row["anomaly_score"]),
        "reason": _reason(row),
        "temperature_c": float(row["temperature_c"]),
        "temp_baseline_c": float(row["temp_baseline_c"]),
        "temp_delta_c": float(row["temp_delta_c"]),
        "thermal_score": float(row["thermal_score"]),
        "tilt_deg": float(row["tilt_deg"]),
        "tilt_baseline_deg": float(row["tilt_baseline_deg"]),
        "tilt_delta_deg": float(row["tilt_delta_deg"]),
        "tilt_score": float(row["tilt_score"]),
        "distance_cm": float(row["distance_cm"]),
        "proximity_score": float(row["proximity_score"]),
        "review_state": "NEW",
        "evidence_path": None,
        "evidence_mode": None,
        "ai_review": None,
    }

    # In demo mode, simulate the rover automatically capturing evidence
    # at the moment the event is generated.
    if demo_evidence_path:
        source_path = Path(demo_evidence_path)
        if source_path.exists():
            destination = EVENT_EVIDENCE_DIR / f"{event_id}_demo.jpg"
            shutil.copy2(source_path, destination)
            event["evidence_path"] = str(destination)
            event["evidence_mode"] = "SIMULATED AUTO-CAPTURE"

    events.append(event)
    save_events(events)
    return event, True


def attach_evidence(event_id, file_bytes, extension="jpg", mode="MANUAL ATTACHMENT"):
    events = load_events()
    extension = extension.lower().replace(".", "") or "jpg"
    destination = EVENT_EVIDENCE_DIR / f"{event_id}.{extension}"
    destination.write_bytes(file_bytes)

    for event in events:
        if event["event_id"] == event_id:
            event["evidence_path"] = str(destination)
            event["evidence_mode"] = mode
            save_events(events)
            return event

    return None


def set_ai_review(event_id, text):
    events = load_events()

    for event in events:
        if event["event_id"] == event_id:
            event["ai_review"] = text
            save_events(events)
            return event

    return None


def set_review_state(event_id, state):
    events = load_events()

    for event in events:
        if event["event_id"] == event_id:
            event["review_state"] = state
            save_events(events)
            return event

    return None


def delete_event(event_id):
    events = load_events()
    kept = []

    for event in events:
        if event["event_id"] == event_id:
            evidence = event.get("evidence_path")
            if evidence:
                path = Path(evidence)
                if path.exists():
                    path.unlink()
        else:
            kept.append(event)

    save_events(kept)


def clear_events():
    save_events([])

    if EVENT_EVIDENCE_DIR.exists():
        for path in EVENT_EVIDENCE_DIR.iterdir():
            if path.is_file():
                path.unlink()
