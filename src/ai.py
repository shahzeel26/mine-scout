from io import BytesIO
import os

from google import genai
from PIL import Image

DEFAULT_MODEL = "gemini-3.8-flash"


def get_api_key(temporary_key=None):
    return (temporary_key or os.getenv("GEMINI_API_KEY") or "").strip()


def get_model():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _client(temporary_key=None):
    api_key = get_api_key(temporary_key)
    if not api_key:
        raise ValueError("No Gemini API key configured.")
    return genai.Client(api_key=api_key)


def _mission_context(df):
    if df.empty:
        return "No mission telemetry is available."

    evaluated = df[df["status"] != "BASELINE"]
    flagged = evaluated[evaluated["status"].isin(["WATCH", "HIGH"])]

    lines = [
        f"Total evaluated points: {len(evaluated)}",
        f"Baseline temperature: {float(df['temp_baseline_c'].iloc[0]):.1f} C",
        f"Baseline tilt: {float(df['tilt_baseline_deg'].iloc[0]):.1f} deg",
        f"Normal points: {int((evaluated['status'] == 'NORMAL').sum())}",
        f"Watch points: {int((evaluated['status'] == 'WATCH').sum())}",
        f"High-priority points: {int((evaluated['status'] == 'HIGH').sum())}",
    ]

    if not flagged.empty:
        lines.append("Flagged observations:")
        for _, row in flagged.iterrows():
            lines.append(
                f"- {row['location']}: temp={float(row['temperature_c']):.1f} C, "
                f"temp_delta={float(row['temp_delta_c']):+.1f} C, "
                f"tilt={float(row['tilt_deg']):.1f} deg, "
                f"distance={float(row['distance_cm']):.1f} cm, "
                f"score={float(row['anomaly_score']):.0f}/100, "
                f"status={row['status']}"
            )

    return "\n".join(lines)


def generate_inspection_brief(
    df,
    mission_name,
    vision_result=None,
    temporary_key=None,
):
    client = _client(temporary_key)
    model = get_model()

    vision_text = "No visual comparison has been performed."
    if vision_result:
        vision_text = (
            f"Local computer vision comparison: "
            f"similarity={vision_result['similarity']:.1f}%, "
            f"changed_area={vision_result['changed_area_pct']:.2f}%, "
            f"highlighted_regions={len(vision_result['boxes'])}."
        )

    prompt = f"""
You are the AI-assisted reporting layer for MineScout, a prototype mining
ground-inspection rover.

Mission: {mission_name}

Telemetry:
{_mission_context(df)}

Visual evidence:
{vision_text}

Write a concise professional inspection brief for a mining engineer.

Requirements:
- Summarise only evidence provided above.
- Separate observation from interpretation.
- Mention the highest-priority inspection points.
- Do not say an area is safe, unsafe, compliant, non-compliant, or diagnose a failure.
- Do not invent measurements, causes, regulations, or recommendations.
- Use wording such as "flagged for expert review" where appropriate.
- Maximum 180 words.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()


def analyze_visual_evidence(
    previous_bytes,
    previous_mime,
    current_bytes,
    current_mime,
    local_result,
    temporary_key=None,
):
    client = _client(temporary_key)
    model = get_model()

    previous_image = Image.open(BytesIO(previous_bytes)).convert("RGB")
    current_image = Image.open(BytesIO(current_bytes)).convert("RGB")

    prompt = f"""
You are reviewing two images from the same MineScout inspection point.

Image 1 is the previous/reference inspection.
Image 2 is the current inspection.

A local computer-vision algorithm measured:
- image similarity: {local_result['similarity']:.1f}%
- changed area: {local_result['changed_area_pct']:.2f}%
- highlighted change regions: {len(local_result['boxes'])}

Return exactly these headings:

OBSERVED CHANGE:
POSSIBLE SIGNIFICANCE:
EXPERT REVIEW FOCUS:

Rules:
- Describe only visible evidence.
- Treat the local change score as supporting information, not ground truth.
- Do not diagnose geotechnical instability, fire, contamination, equipment failure,
  or safety status.
- Do not claim compliance.
- If viewpoint or lighting makes comparison unreliable, say so clearly.
- Maximum 120 words.
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt, previous_image, current_image],
    )
    return (response.text or "").strip()


def generate_event_review(event, temporary_key=None):
    client = _client(temporary_key)
    model = get_model()

    prompt = f"""
You are MineScout's AI-assisted event reviewer.

Review this inspection event:

Event ID: {event['event_id']}
Mission: {event['mission_name']}
Point: {event['point']}
Classification: {event['classification']}
Combined anomaly score: {event['anomaly_score']:.0f}/100

Temperature: {event['temperature_c']:.1f} C
Temperature baseline: {event['temp_baseline_c']:.1f} C
Temperature deviation: {event['temp_delta_c']:+.1f} C
Thermal score: {event['thermal_score']:.0f}/100

Tilt: {event['tilt_deg']:.1f} deg
Tilt baseline: {event['tilt_baseline_deg']:.1f} deg
Tilt deviation: {event['tilt_delta_deg']:+.1f} deg
Tilt score: {event['tilt_score']:.0f}/100

Obstacle distance: {event['distance_cm']:.1f} cm
Proximity score: {event['proximity_score']:.0f}/100

System reason: {event['reason']}
Evidence mode: {event.get('evidence_mode') or 'No image evidence attached'}

Write exactly:

EVENT SUMMARY:
WHY IT WAS FLAGGED:
EXPERT REVIEW FOCUS:

Rules:
- Use only the measurements above.
- Do not diagnose a cause.
- Do not label the location safe or unsafe.
- Do not make a compliance determination.
- Do not invent thresholds or regulations.
- Maximum 120 words.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()


def ask_minescout(
    question,
    df,
    mission_name,
    vision_result=None,
    events=None,
    temporary_key=None,
):
    client = _client(temporary_key)
    model = get_model()

    vision_text = (
        "No visual comparison available."
        if not vision_result
        else (
            f"Visual comparison: similarity {vision_result['similarity']:.1f}%, "
            f"changed area {vision_result['changed_area_pct']:.2f}%, "
            f"{len(vision_result['boxes'])} highlighted regions."
        )
    )

    event_text = "No inspection events recorded."
    if events:
        lines = []
        for event in events:
            lines.append(
                f"- {event['event_id']} at {event['point']}: "
                f"{event['classification']}, score {event['anomaly_score']:.0f}/100, "
                f"review_state={event['review_state']}, reason={event['reason']}"
            )
        event_text = "\n".join(lines)

    prompt = f"""
You are MineScout Copilot. Answer only from the current mission evidence.

Mission: {mission_name}

Telemetry:
{_mission_context(df)}

Visual evidence:
{vision_text}

Inspection events:
{event_text}

User question:
{question}

Rules:
- If the evidence cannot answer the question, say so.
- Never invent sensor readings, unseen conditions, causes, or regulations.
- Never provide a safety certification or engineering determination.
- Keep the answer concise and operational.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()
