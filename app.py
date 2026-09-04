from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.anomaly import enrich_mission
from src.ai import (
    analyze_visual_evidence,
    ask_minescout,
    generate_event_review,
    generate_inspection_brief,
    get_api_key,
    get_model,
)
from src.events import (
    attach_evidence,
    clear_events,
    delete_event,
    ensure_sensor_event,
    load_events,
    set_ai_review,
    set_review_state,
)
from src.reporting import build_structured_report
from src.simulator import simulate_mission
from src.vision import compare_images


load_dotenv()

DATA_DIR = Path("data")
DEMO_FILE = DATA_DIR / "demo_mission.csv"
LIVE_FILE = DATA_DIR / "live_sensor_readings.csv"
DEMO_EVIDENCE = Path("evidence/demo_current.jpg")

DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="MineScout Operations",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1520px;
}
.hero {
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 20px;
    padding: 1.4rem 1.55rem;
    background:
        radial-gradient(circle at 5% 5%, rgba(242,184,75,.12), transparent 28%),
        linear-gradient(135deg, rgba(255,255,255,.025), rgba(255,255,255,.005));
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0 0 .3rem 0;
    font-size: 2.2rem;
}
.hero p {
    margin: 0;
    opacity: .72;
}
.eyebrow {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    opacity: .6;
    font-weight: 700;
}
.statusbar {
    border-radius: 13px;
    padding: .8rem 1rem;
    border: 1px solid rgba(255,255,255,.08);
    margin-bottom: .8rem;
    font-weight: 700;
}
.normal { background: rgba(42,154,82,.12); }
.watch { background: rgba(242,184,75,.12); }
.high { background: rgba(215,72,72,.13); }
.event-card {
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    background: rgba(255,255,255,.018);
}
.event-new {
    border-left: 4px solid #F2B84B;
}
.event-reviewed {
    border-left: 4px solid #49A36C;
}
.small {
    font-size: .82rem;
    opacity: .68;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(255,255,255,.075);
    background: rgba(255,255,255,.016);
    padding: .85rem 1rem;
    border-radius: 15px;
}
div[data-testid="stTabs"] button {
    font-weight: 650;
}
</style>
""",
    unsafe_allow_html=True,
)

if not DEMO_FILE.exists():
    simulate_mission().to_csv(DEMO_FILE, index=False)

defaults = {
    "cursor": 7,
    "vision_result": None,
    "vision_files": None,
    "visual_ai": "",
    "ai_brief": "",
    "copilot_answer": "",
    "event_notice": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.markdown(
    """
<div class="hero">
    <div class="eyebrow">WADSIH Hardware Hack 2026 · Mining & Industrial</div>
    <h1>MineScout Operations</h1>
    <p>Remote inspection evidence · multi-sensor anomaly intelligence · AI-assisted expert review</p>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Mission control")

    mission_name = st.text_input(
        "Mission name",
        "Restricted Waste-Dump Inspection",
    )

    source = st.radio(
        "Telemetry source",
        ["Demo mission", "Live rover"],
        help="Live rover reads telemetry received by backend_receiver.py.",
    )

    baseline_points = st.slider(
        "Calibration observations",
        3,
        8,
        5,
        help="Early observations establish robust mission-specific baselines.",
    )

    if source == "Demo mission":
        st.markdown("### Demo playback")
        demo_df = pd.read_csv(DEMO_FILE)
        max_points = len(demo_df)

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Reset", use_container_width=True):
                st.session_state.cursor = max(0, baseline_points - 1)
                st.session_state.event_notice = ""

        with c2:
            if st.button("Advance", use_container_width=True):
                st.session_state.cursor = min(
                    st.session_state.cursor + 1,
                    max_points - 1,
                )

        st.session_state.cursor = st.slider(
            "Mission progress",
            0,
            max_points - 1,
            min(st.session_state.cursor, max_points - 1),
        )

    st.divider()
    st.markdown("### Gemini")

    temporary_key = st.text_input(
        "Temporary API key",
        type="password",
        help="Prefer GEMINI_API_KEY in .env. This temporary value is not written to disk.",
    )

    if get_api_key(temporary_key):
        st.success(f"AI ready · {get_model()}")
    else:
        st.caption("AI features are disabled until a Gemini key is configured.")

    st.divider()

    if st.button("Clear inspection event history", use_container_width=True):
        clear_events()
        st.session_state.event_notice = ""
        st.success("Event history cleared.")


# ---------- LOAD MISSION DATA ----------
if source == "Live rover":
    if LIVE_FILE.exists() and LIVE_FILE.stat().st_size > 0:
        raw = pd.read_csv(LIVE_FILE)
        source_label = "LIVE ROVER"
        is_demo = False
    else:
        raw = pd.read_csv(DEMO_FILE).iloc[: st.session_state.cursor + 1].copy()
        source_label = "DEMO FALLBACK"
        is_demo = True
        st.warning(
            "No live rover telemetry has arrived yet. Showing demo telemetry instead."
        )
else:
    raw = pd.read_csv(DEMO_FILE).iloc[: st.session_state.cursor + 1].copy()
    source_label = "DEMO"
    is_demo = True

df = enrich_mission(raw, baseline_points=baseline_points)

evaluated = df[df["status"] != "BASELINE"]
flagged = evaluated[evaluated["status"].isin(["WATCH", "HIGH"])]

high_count = int((evaluated["status"] == "HIGH").sum())
watch_count = int((evaluated["status"] == "WATCH").sum())

if high_count:
    mission_status = "REVIEW REQUIRED"
    status_css = "high"
elif watch_count:
    mission_status = "WATCH"
    status_css = "watch"
else:
    mission_status = "NORMAL"
    status_css = "normal"

latest = df.iloc[-1]

# ---------- AUTOMATIC EVENT CREATION ----------
new_event, created = ensure_sensor_event(
    mission_name=mission_name,
    row=latest,
    source=source_label,
    demo_evidence_path=DEMO_EVIDENCE if is_demo else None,
)

if created and new_event:
    st.session_state.event_notice = (
        f"{new_event['event_id']} automatically created at "
        f"{new_event['point']} · {new_event['classification']} · "
        f"{new_event['anomaly_score']:.0f}/100"
    )

all_events = load_events()
mission_events = [
    event for event in all_events
    if event.get("mission_name") == mission_name
]

new_events = [
    event for event in mission_events
    if event.get("review_state") == "NEW"
]


st.markdown(
    f"""
<div class="statusbar {status_css}">
Mission status: {mission_status}
&nbsp; · &nbsp; Source: {source_label}
&nbsp; · &nbsp; Current point: {latest['location']}
&nbsp; · &nbsp; Event engine: ACTIVE
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.event_notice:
    st.toast(st.session_state.event_notice)


# ---------- TOP METRICS ----------
m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(
    "Baseline temp.",
    f"{float(df['temp_baseline_c'].iloc[0]):.1f} °C",
)
m2.metric(
    "Current temp.",
    f"{float(latest['temperature_c']):.1f} °C",
    f"{float(latest['temp_delta_c']):+.1f} °C",
)
m3.metric(
    "Current tilt",
    f"{float(latest['tilt_deg']):.1f}°",
)
m4.metric(
    "Obstacle range",
    f"{float(latest['distance_cm']):.0f} cm",
)
m5.metric(
    "Anomaly score",
    f"{float(latest['anomaly_score']):.0f}/100",
)
m6.metric(
    "Open events",
    len(new_events),
)


tabs = st.tabs(
    [
        "Operations",
        "Event Centre",
        "Evidence & Vision",
        "AI Copilot",
        "Inspection Record",
        "Data",
    ]
)


# =====================================================================
# OPERATIONS
# =====================================================================
with tabs[0]:
    chart_col, point_col = st.columns([1.7, 1])

    with chart_col:
        st.subheader("Mission telemetry")

        trend = df[
            ["timestamp", "temperature_c", "temp_baseline_c"]
        ].copy()

        trend["timestamp"] = pd.to_datetime(trend["timestamp"])

        trend = trend.rename(
            columns={
                "temperature_c": "Observed temperature",
                "temp_baseline_c": "Mission baseline",
            }
        )

        long_trend = trend.melt(
            id_vars=["timestamp"],
            value_vars=["Observed temperature", "Mission baseline"],
            var_name="Series",
            value_name="Temperature (°C)",
        )

        fig = px.line(
            long_trend,
            x="timestamp",
            y="Temperature (°C)",
            color="Series",
            markers=True,
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with point_col:
        st.subheader("Current inspection point")

        st.markdown(
            f"""
<div class="event-card">
    <div class="eyebrow">Point</div>
    <h2>{latest['location']}</h2>
    <div class="eyebrow">Classification</div>
    <h3>{latest['status']}</h3>
    <div class="eyebrow">Combined anomaly score</div>
    <h2>{float(latest['anomaly_score']):.0f} / 100</h2>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(
            min(int(round(float(latest["anomaly_score"]))), 100),
            text="Current anomaly severity",
        )

        if str(latest.get("connection", "UNKNOWN")) == "CONNECTED":
            st.success("Rover telemetry link: CONNECTED")
        else:
            st.error(
                "Telemetry link lost — rover hardware should enter its configured safe state."
            )

    st.subheader("Why this score?")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "Thermal contribution",
            f"{float(latest['thermal_score']):.0f}/100",
            f"{float(latest['temp_delta_c']):+.1f} °C vs baseline",
        )
        st.progress(
            min(int(float(latest["thermal_score"])), 100),
            text="70% model weight",
        )

    with s2:
        st.metric(
            "Tilt contribution",
            f"{float(latest['tilt_score']):.0f}/100",
            f"{float(latest['tilt_delta_deg']):+.1f}° vs baseline",
        )
        st.progress(
            min(int(float(latest["tilt_score"])), 100),
            text="20% model weight",
        )

    with s3:
        st.metric(
            "Proximity contribution",
            f"{float(latest['proximity_score']):.0f}/100",
            f"{float(latest['distance_cm']):.0f} cm range",
        )
        st.progress(
            min(int(float(latest["proximity_score"])), 100),
            text="10% model weight",
        )

    st.caption(
        "The weights are prototype prioritisation weights, not validated mining safety limits. "
        "They should be calibrated for a specific asset and inspection task before deployment."
    )

    st.subheader("Mission timeline")

    timeline_cols = [
        "timestamp",
        "location",
        "temperature_c",
        "temp_delta_c",
        "tilt_deg",
        "distance_cm",
        "thermal_score",
        "tilt_score",
        "proximity_score",
        "anomaly_score",
        "status",
    ]

    st.dataframe(
        df[timeline_cols].tail(12),
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# EVENT CENTRE
# =====================================================================
with tabs[1]:
    st.subheader("Inspection Event Centre")
    st.caption(
        "MineScout automatically creates an event when a sensor observation reaches WATCH or HIGH. "
        "The event preserves the measurements, evidence and review history."
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total events", len(mission_events))
    e2.metric("New", len(new_events))
    e3.metric(
        "Reviewed",
        sum(event.get("review_state") == "REVIEWED" for event in mission_events),
    )
    e4.metric(
        "With evidence",
        sum(bool(event.get("evidence_path")) for event in mission_events),
    )

    if not mission_events:
        st.info(
            "No events yet. In demo mode, advance the mission until an anomalous point is reached."
        )
    else:
        for event in reversed(mission_events):
            title = (
                f"{event['event_id']} · {event['point']} · "
                f"{event['classification']} · {event['anomaly_score']:.0f}/100"
            )

            with st.expander(title, expanded=event["review_state"] == "NEW"):
                card_class = (
                    "event-reviewed"
                    if event["review_state"] == "REVIEWED"
                    else "event-new"
                )

                st.markdown(
                    f"""
<div class="event-card {card_class}">
    <div class="eyebrow">Event reason</div>
    <h3>{event['reason']}</h3>
    <div class="small">
        Created {event['created_at']} · Source {event['source']} ·
        Review state {event['review_state']}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.write("")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Temperature",
                    f"{event['temperature_c']:.1f} °C",
                    f"{event['temp_delta_c']:+.1f} °C",
                )
                c2.metric(
                    "Tilt",
                    f"{event['tilt_deg']:.1f}°",
                    f"{event['tilt_delta_deg']:+.1f}°",
                )
                c3.metric(
                    "Range",
                    f"{event['distance_cm']:.0f} cm",
                )
                c4.metric(
                    "Combined score",
                    f"{event['anomaly_score']:.0f}/100",
                )

                left, right = st.columns([1.1, 1])

                with left:
                    evidence_path = event.get("evidence_path")

                    if evidence_path and Path(evidence_path).exists():
                        st.image(
                            evidence_path,
                            caption=(
                                f"{event.get('evidence_mode') or 'Inspection evidence'} · "
                                f"{event['event_id']}"
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("No image evidence attached to this event.")

                    uploaded = st.file_uploader(
                        "Attach / replace event evidence",
                        type=["jpg", "jpeg", "png", "webp"],
                        key=f"event_upload_{event['event_id']}",
                    )

                    if uploaded and st.button(
                        "Save evidence",
                        key=f"save_evidence_{event['event_id']}",
                    ):
                        suffix = Path(uploaded.name).suffix.lstrip(".") or "jpg"
                        attach_evidence(
                            event["event_id"],
                            uploaded.getvalue(),
                            extension=suffix,
                            mode="MANUAL EVENT EVIDENCE",
                        )
                        st.success("Evidence attached. Rerun or reopen the event to view it.")

                with right:
                    st.markdown("#### AI-assisted event review")

                    if event.get("ai_review"):
                        st.write(event["ai_review"])
                    else:
                        st.caption("No AI event review generated yet.")

                    if get_api_key(temporary_key):
                        if st.button(
                            "Generate Gemini event review",
                            key=f"ai_event_{event['event_id']}",
                        ):
                            with st.spinner("Reviewing event evidence..."):
                                try:
                                    review = generate_event_review(
                                        event,
                                        temporary_key=temporary_key,
                                    )
                                    set_ai_review(event["event_id"], review)
                                    st.success("AI-assisted review saved to event history.")
                                    st.write(review)
                                except Exception as exc:
                                    st.error(f"Gemini request failed: {exc}")

                    action1, action2 = st.columns(2)

                    with action1:
                        if event["review_state"] == "NEW":
                            if st.button(
                                "Mark reviewed",
                                key=f"review_{event['event_id']}",
                                use_container_width=True,
                            ):
                                set_review_state(
                                    event["event_id"],
                                    "REVIEWED",
                                )
                                st.success("Event marked reviewed.")
                        else:
                            if st.button(
                                "Reopen",
                                key=f"reopen_{event['event_id']}",
                                use_container_width=True,
                            ):
                                set_review_state(
                                    event["event_id"],
                                    "NEW",
                                )
                                st.success("Event reopened.")

                    with action2:
                        if st.button(
                            "Delete event",
                            key=f"delete_{event['event_id']}",
                            use_container_width=True,
                        ):
                            delete_event(event["event_id"])
                            st.warning("Event deleted.")


# =====================================================================
# EVIDENCE & VISION
# =====================================================================
with tabs[2]:
    st.subheader("Repeat-inspection visual comparison")
    st.caption(
        "Compare a previous inspection image with the current inspection. "
        "Local OpenCV/SSIM analysis works offline; Gemini review is optional."
    )

    up1, up2 = st.columns(2)

    with up1:
        previous_file = st.file_uploader(
            "Previous / reference image",
            type=["jpg", "jpeg", "png", "webp"],
            key="previous_image",
        )
        if previous_file:
            st.image(
                previous_file,
                caption="Reference inspection",
                use_container_width=True,
            )

    with up2:
        current_file = st.file_uploader(
            "Current inspection image",
            type=["jpg", "jpeg", "png", "webp"],
            key="current_image",
        )

        camera_file = st.camera_input(
            "Or capture current evidence",
            key="current_camera",
        )

        selected_current = current_file or camera_file

        if selected_current:
            st.image(
                selected_current,
                caption="Current inspection",
                use_container_width=True,
            )

    if previous_file and selected_current:
        if st.button(
            "Compare inspection images",
            type="primary",
        ):
            previous_bytes = previous_file.getvalue()
            current_bytes = selected_current.getvalue()

            result = compare_images(
                previous_bytes,
                current_bytes,
            )

            st.session_state.vision_result = result
            st.session_state.vision_files = {
                "previous_bytes": previous_bytes,
                "previous_mime": previous_file.type or "image/jpeg",
                "current_bytes": current_bytes,
                "current_mime": selected_current.type or "image/jpeg",
            }
            st.session_state.visual_ai = ""

    vision = st.session_state.vision_result

    if vision:
        st.divider()

        v1, v2, v3 = st.columns(3)
        v1.metric(
            "Image similarity",
            f"{vision['similarity']:.1f}%",
        )
        v2.metric(
            "Changed area",
            f"{vision['changed_area_pct']:.2f}%",
        )
        v3.metric(
            "Change regions",
            len(vision["boxes"]),
        )

        img1, img2 = st.columns(2)

        with img1:
            st.image(
                vision["annotated_rgb"],
                caption="Current image · highlighted change regions",
                use_container_width=True,
            )

        with img2:
            st.image(
                vision["difference_mask"],
                caption="Local CV difference mask",
                use_container_width=True,
                clamp=True,
            )

        if st.session_state.vision_files and get_api_key(temporary_key):
            if st.button("Ask Gemini to review both images"):
                files = st.session_state.vision_files

                with st.spinner("Gemini is reviewing visual evidence..."):
                    try:
                        st.session_state.visual_ai = analyze_visual_evidence(
                            files["previous_bytes"],
                            files["previous_mime"],
                            files["current_bytes"],
                            files["current_mime"],
                            vision,
                            temporary_key=temporary_key,
                        )
                    except Exception as exc:
                        st.error(f"Gemini request failed: {exc}")

        if st.session_state.visual_ai:
            st.markdown("### Gemini visual review")
            st.write(st.session_state.visual_ai)


# =====================================================================
# AI COPILOT
# =====================================================================
with tabs[3]:
    st.subheader("MineScout AI Copilot")

    if not get_api_key(temporary_key):
        st.info(
            "Add GEMINI_API_KEY to .env or provide a temporary key in the sidebar."
        )
    else:
        st.caption(
            f"Model: {get_model()} · AI responses are grounded in current telemetry "
            "and the recorded MineScout event history."
        )

        if st.button(
            "Generate mission inspection brief",
            type="primary",
        ):
            with st.spinner("Generating inspection brief..."):
                try:
                    st.session_state.ai_brief = generate_inspection_brief(
                        df,
                        mission_name,
                        vision_result=st.session_state.vision_result,
                        temporary_key=temporary_key,
                    )
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")

        if st.session_state.ai_brief:
            st.markdown("### Mission brief")
            st.write(st.session_state.ai_brief)

        st.divider()

        question = st.text_input(
            "Ask about the current mission",
            placeholder="Which event should the engineer review first, and what evidence triggered it?",
        )

        if st.button("Ask MineScout") and question.strip():
            with st.spinner("Reviewing mission evidence..."):
                try:
                    st.session_state.copilot_answer = ask_minescout(
                        question,
                        df,
                        mission_name,
                        vision_result=st.session_state.vision_result,
                        events=mission_events,
                        temporary_key=temporary_key,
                    )
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")

        if st.session_state.copilot_answer:
            st.markdown("### Copilot response")
            st.write(st.session_state.copilot_answer)


# =====================================================================
# INSPECTION RECORD
# =====================================================================
with tabs[4]:
    st.subheader("Structured inspection record")
    st.caption(
        "This report combines sensor analytics, event history, visual evidence metrics "
        "and optional AI-assisted summaries."
    )

    report = build_structured_report(
        df,
        mission_name,
        events=mission_events,
        vision_result=st.session_state.vision_result,
        ai_brief=st.session_state.ai_brief or None,
    )

    st.markdown(report)

    st.download_button(
        "Download inspection record (.md)",
        data=report,
        file_name="minescout_inspection_record.md",
        mime="text/markdown",
    )


# =====================================================================
# DATA
# =====================================================================
with tabs[5]:
    st.subheader("Processed telemetry")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download telemetry CSV",
        data=df.to_csv(index=False),
        file_name="minescout_processed_telemetry.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Event store")

    if mission_events:
        event_df = pd.DataFrame(mission_events)
        st.dataframe(
            event_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No events stored for this mission yet.")

    st.caption(
        "Live integration path: ESP32 → backend_receiver.py → "
        "data/live_sensor_readings.csv → anomaly engine → automatic event creation."
    )
