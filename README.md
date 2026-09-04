# MineScout V4 — Automatic Inspection Events

V4 moves MineScout from a dashboard into an inspection-event workflow.

## Core idea

When a rover observation becomes unusual, MineScout should not only show a high score.
It should automatically preserve the evidence as an inspection event that can be reviewed later.

```text
Rover observation
      ↓
Multi-sensor anomaly engine
      ↓
WATCH / HIGH
      ↓
Automatic inspection event
      ↓
Sensor snapshot + evidence image
      ↓
AI-assisted review
      ↓
Human review state
      ↓
Structured inspection record
```

## New in V4

- Automatic event creation when an observation reaches WATCH or HIGH
- Persistent event history in `data/events.json`
- Event IDs such as `EVT-001`
- Sensor snapshot saved with each event
- Explicit reason for why the event was created
- Event review states: `NEW` / `REVIEWED`
- Event evidence attachment
- Simulated automatic image capture during demo missions
- Per-event Gemini review
- Improved score explainability:
  - thermal score
  - tilt score
  - proximity score
  - weighted combined score
- Event log included in the final inspection record
- Updated operations UI with an Event Centre

## Demo behaviour

The demo mission contains deliberately unusual observations.

When you press **Advance** and reach an unusual point:

1. the anomaly engine scores the observation;
2. MineScout creates an event automatically;
3. the demo copies a simulated camera image into the event;
4. the event appears in **Event Centre**;
5. you can ask Gemini to create a short event review;
6. you can mark the event as reviewed.

The image auto-capture is labelled `SIMULATED AUTO-CAPTURE` so it is not confused with real rover evidence.

## Running the app

From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Gemini configuration

Create a `.env` file next to `app.py`:

```text
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-3.8-flash
```

Do not commit the real key.

The app uses the official `google-genai` SDK with `client.models.generate_content`.

## Recommended V4 test

1. Open **Operations**.
2. Press **Reset**.
3. Press **Advance** until an unusual observation occurs.
4. Confirm an `EVT-...` event appears automatically.
5. Open **Event Centre**.
6. Inspect the sensor breakdown and auto-captured demo evidence.
7. Click **Generate Gemini event review**.
8. Click **Mark reviewed**.
9. Open **Inspection Record** and confirm that the event appears in the report.

## Live rover integration

Start the telemetry API:

```powershell
python -m uvicorn backend_receiver:app --host 0.0.0.0 --port 8000
```

The rover can send:

```json
{
  "location": "Point_07",
  "temperature_c": 31.4,
  "distance_cm": 72.0,
  "tilt_deg": 3.2,
  "connection": "CONNECTED"
}
```

to:

```text
http://YOUR-LAPTOP-IP:8000/telemetry
```

Select **Live rover** in the dashboard.

## Important prototype limitation

The current multi-sensor weights are prototype prioritisation weights:

- thermal: 70%
- tilt: 20%
- proximity: 10%

They are not validated mining safety thresholds. A real deployment would need asset-specific
calibration, sensor validation, environmental testing and domain-expert acceptance criteria.

MineScout is an evidence and prioritisation platform. It does not make autonomous safety,
engineering or compliance determinations.
