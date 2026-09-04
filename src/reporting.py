from datetime import datetime


def build_structured_report(
    df,
    mission_name,
    events=None,
    vision_result=None,
    ai_brief=None,
):
    evaluated = df[df["status"] != "BASELINE"] if not df.empty else df
    flagged = evaluated[evaluated["status"].isin(["WATCH", "HIGH"])]

    high_count = (
        int((evaluated["status"] == "HIGH").sum())
        if not evaluated.empty else 0
    )
    watch_count = (
        int((evaluated["status"] == "WATCH").sum())
        if not evaluated.empty else 0
    )
    normal_count = (
        int((evaluated["status"] == "NORMAL").sum())
        if not evaluated.empty else 0
    )

    mission_status = (
        "REVIEW REQUIRED"
        if high_count > 0
        else "WATCH"
        if watch_count > 0
        else "NO SIGNIFICANT ANOMALIES"
    )

    lines = [
        "# MineScout Inspection Record",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Mission:** {mission_name}",
        f"**Mission status:** {mission_status}",
        "",
        "## Mission summary",
        f"- Evaluated points: {len(evaluated)}",
        f"- Normal: {normal_count}",
        f"- Watch: {watch_count}",
        f"- High-priority: {high_count}",
        f"- Inspection events: {len(events or [])}",
        "",
        "## Flagged sensor observations",
    ]

    if flagged.empty:
        lines.append("- No sensor observations were flagged.")
    else:
        for _, row in flagged.iterrows():
            lines.append(
                f"- **{row['location']}** — temperature "
                f"{float(row['temperature_c']):.1f} °C "
                f"({float(row['temp_delta_c']):+.1f} °C vs baseline), "
                f"tilt {float(row['tilt_deg']):.1f}°, "
                f"combined anomaly score "
                f"{float(row['anomaly_score']):.0f}/100 "
                f"({row['status']})."
            )

    if events:
        lines += ["", "## Inspection event log"]

        for event in events:
            lines += [
                f"### {event['event_id']} · {event['point']}",
                f"- Classification: {event['classification']}",
                f"- Score: {event['anomaly_score']:.0f}/100",
                f"- Reason: {event['reason']}",
                f"- Review state: {event['review_state']}",
                f"- Evidence: {event.get('evidence_mode') or 'None attached'}",
            ]

            if event.get("ai_review"):
                lines.append(
                    f"- AI-assisted review: {event['ai_review'].replace(chr(10), ' ')}"
                )

    if vision_result:
        lines += [
            "",
            "## Visual change evidence",
            f"- Image similarity: {vision_result['similarity']:.1f}%",
            f"- Changed image area: {vision_result['changed_area_pct']:.2f}%",
            f"- Change regions highlighted: {len(vision_result['boxes'])}",
        ]

    if ai_brief:
        lines += [
            "",
            "## AI-assisted mission brief",
            ai_brief.strip(),
        ]

    lines += [
        "",
        "## Review note",
        "MineScout provides inspection evidence and anomaly prioritisation "
        "for expert review. It does not make an autonomous engineering, "
        "compliance, or safety determination.",
    ]

    return "\n".join(lines)
