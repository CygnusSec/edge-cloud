"""
Streamlit Dashboard for Edge-Cloud Demo.

Three pages:
  1. Single Image Analysis  — upload image, choose scenario, display bounding boxes
  2. Scenario Comparison    — summary table + 4 charts from CSV results
  3. Video Analysis         — upload video, process frames via edge-cloud mechanism

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLOUD_URL = os.environ.get("CLOUD_URL", "http://localhost:8000")
EDGE_URL = os.environ.get("EDGE_URL", "http://localhost:8001")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")

SCENARIO_FILES = {
    "cloud_only": os.path.join(RESULTS_DIR, "cloud_only.csv"),
    "edge_only": os.path.join(RESULTS_DIR, "edge_only.csv"),
    "edge_cloud": os.path.join(RESULTS_DIR, "edge_cloud.csv"),
}

SCENARIO_LABELS = {
    "cloud_only": "Cloud-Only",
    "edge_only": "Edge-Only",
    "edge_cloud": "Edge-Cloud",
}

COLORS = {
    "cloud_only": "#EF553B",
    "edge_only": "#00CC96",
    "edge_cloud": "#636EFA",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_csv(path: str) -> pd.DataFrame | None:
    """Load a results CSV, skipping the SUMMARY row. Returns None if missing."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df = df[~df.iloc[:, 0].astype(str).str.startswith("SUMMARY")]
        if df.empty:
            return None
        return df
    except Exception:
        return None


def draw_bounding_boxes(image_bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Draw bounding boxes with labels on a BGR image."""
    img = image_bgr.copy()
    for obj in objects:
        bbox = obj.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        label = obj.get("class_name", "?")
        conf = obj.get("confidence", 0.0)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def call_cloud(image_bytes: bytes, filename: str = "image.jpg") -> dict | None:
    """Call Cloud Node POST /predict. Returns parsed JSON or None on error."""
    try:
        response = requests.post(
            f"{CLOUD_URL}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as exc:
        st.error(f"Cloud Node connection error: {exc}")
    return None


def call_edge(image_bytes: bytes, filename: str = "image.jpg") -> dict | None:
    """
    Call Edge Node API POST /predict using YOLOv8n (lightweight model).
    Returns parsed JSON with processing_place='edge', or None on error.
    """
    try:
        response = requests.post(
            f"{EDGE_URL}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as exc:
        st.error(f"Edge Node connection error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Page 1: Single Image Analysis
# ---------------------------------------------------------------------------


def _should_keep_at_edge(edge_result: dict, threshold: float, threshold_avg: float, object_threshold: int) -> tuple[bool, dict]:
    """
    Combined offload decision:
      Keep at Edge if ALL three conditions are met:
        1. max_confidence >= threshold
        2. average_confidence >= threshold_avg
        3. object_count <= object_threshold
    Returns (keep_at_edge, metrics_dict)
    """
    objects = edge_result.get("objects", [])
    object_count = edge_result.get("object_count", len(objects))
    confs = [o.get("confidence", 0.0) for o in objects] if objects else []

    max_conf = max(confs) if confs else 0.0
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    cond1 = max_conf >= threshold
    cond2 = avg_conf >= threshold_avg
    cond3 = object_count <= object_threshold

    keep = cond1 and cond2 and cond3

    return keep, {
        "max_conf": max_conf,
        "avg_conf": avg_conf,
        "object_count": object_count,
        "cond1": cond1,
        "cond2": cond2,
        "cond3": cond3,
    }


def page_single_image(threshold: float = CONFIDENCE_THRESHOLD, threshold_avg: float = 0.4, object_threshold: int = 10):
    st.header("🖼️ Single Image Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload image (JPEG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_image_uploader",
            accept_multiple_files=False,
        )
    with col2:
        scenario = st.selectbox(
            "Select scenario",
            options=["edge_cloud", "cloud_only", "edge_only"],
            format_func=lambda x: SCENARIO_LABELS[x],
        )
        if scenario == "edge_cloud":
            st.info(
                f"Current thresholds: max ≥ **{threshold:.2f}**, avg ≥ **{threshold_avg:.2f}**, objects ≤ **{object_threshold}**\n\n"
                "Adjust in sidebar ←"
            )

    if uploaded is None:
        st.info("Please upload an image to start analysis.")
        return

    image_bytes = uploaded.read()
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Cannot read image. Please try a different file.")
        return

    result = None
    processing_place = "—"
    edge_conf_used = None
    edge_result_saved = None
    decision_metrics = None

    with st.spinner("Analyzing..."):
        if scenario == "cloud_only":
            result = call_cloud(image_bytes, uploaded.name)
            if result:
                processing_place = "☁️ Cloud"

        elif scenario == "edge_only":
            result = call_edge(image_bytes)
            if result:
                processing_place = "📱 Edge"

        else:  # edge_cloud
            edge_result = call_edge(image_bytes)
            if edge_result:
                edge_result_saved = edge_result
                keep, metrics = _should_keep_at_edge(edge_result, threshold, threshold_avg, object_threshold)
                decision_metrics = metrics
                edge_conf_used = metrics["max_conf"]

                if keep:
                    result = edge_result
                    processing_place = "📱 Edge"
                else:
                    result = call_cloud(image_bytes, uploaded.name)
                    if result:
                        processing_place = "☁️ Cloud"
            else:
                result = call_cloud(image_bytes, uploaded.name)
                if result:
                    processing_place = "☁️ Cloud"

    if result is None:
        st.error("Analysis failed.")
        return

    annotated = draw_bounding_boxes(image_bgr, result.get("objects", []))
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detection result", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed at", processing_place)
    m2.metric("Confidence", f"{result.get('confidence', 0):.3f}")
    m3.metric("Latency", f"{result.get('latency_ms', 0):.1f} ms")
    m4.metric("Objects detected", result.get("object_count", 0))

    if result.get("detected_classes"):
        st.write("**Detected classes:**", ", ".join(result["detected_classes"]))

    # Show edge-cloud decision explanation
    if scenario == "edge_cloud" and decision_metrics is not None:
        final_conf = result.get("confidence", 0) if result else 0
        m = decision_metrics
        st.markdown("---")
        st.markdown("**Edge-Cloud Decision Logic:**")

        # Show 3 conditions
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "① Max confidence",
            f"{m['max_conf']:.3f}",
            delta=f"{'✅' if m['cond1'] else '❌'} threshold {threshold:.2f}",
        )
        c2.metric(
            "② Avg confidence",
            f"{m['avg_conf']:.3f}",
            delta=f"{'✅' if m['cond2'] else '❌'} threshold_avg {threshold_avg:.2f}",
        )
        c3.metric(
            "③ Object count",
            m['object_count'],
            delta=f"{'✅' if m['cond3'] else '❌'} max {object_threshold}",
        )

        if processing_place.startswith("📱"):
            st.success(
                f"All 3 conditions met → **Keep at Edge** (0 bytes uploaded)\n\n"
                f"① max_conf {m['max_conf']:.3f} ≥ {threshold:.2f}  "
                f"② avg_conf {m['avg_conf']:.3f} ≥ {threshold_avg:.2f}  "
                f"③ objects {m['object_count']} ≤ {object_threshold}"
            )
        else:
            failed = []
            if not m['cond1']: failed.append(f"max_conf {m['max_conf']:.3f} < {threshold:.2f}")
            if not m['cond2']: failed.append(f"avg_conf {m['avg_conf']:.3f} < {threshold_avg:.2f}")
            if not m['cond3']: failed.append(f"objects {m['object_count']} > {object_threshold}")
            st.warning(
                f"Condition(s) failed → **Offload to Cloud** (+{len(image_bytes):,} bytes)\n\n"
                f"Failed: {' | '.join(failed)}\n\n"
                f"Cloud result: confidence **{final_conf:.3f}** ({'+' if final_conf > m['max_conf'] else ''}{final_conf - m['max_conf']:.3f} vs edge max)"
            )

        # ── Side-by-side Edge vs Cloud comparison ──────────────────────────
        if edge_result_saved is not None and result is not None and processing_place.startswith("☁️"):
            st.markdown("---")
            st.markdown("### 🔄 Edge vs Cloud — Side-by-Side Comparison")
            st.caption("Both models ran on the same image. Edge result triggered offload; Cloud result is used as final output.")

            col_edge, col_cloud = st.columns(2)

            # Draw edge bounding boxes
            edge_annotated = draw_bounding_boxes(image_bgr, edge_result_saved.get("objects", []))
            edge_rgb = cv2.cvtColor(edge_annotated, cv2.COLOR_BGR2RGB)

            # Draw cloud bounding boxes
            cloud_annotated = draw_bounding_boxes(image_bgr, result.get("objects", []))
            cloud_rgb = cv2.cvtColor(cloud_annotated, cv2.COLOR_BGR2RGB)

            with col_edge:
                st.markdown("#### 📱 Edge (YOLOv8n) — *not used*")
                st.image(edge_rgb, use_container_width=True)
                e_conf = edge_result_saved.get("confidence", 0)
                e_count = edge_result_saved.get("object_count", 0)
                e_lat = edge_result_saved.get("latency_ms", 0)
                e_classes = edge_result_saved.get("detected_classes", [])
                st.metric("Confidence", f"{e_conf:.3f}", delta=f"{e_conf - final_conf:.3f} vs cloud")
                st.metric("Objects", e_count)
                st.metric("Latency", f"{e_lat:.1f} ms")
                st.metric("Upload", "0 bytes")
                if e_classes:
                    st.caption(f"Classes: {', '.join(e_classes)}")

            with col_cloud:
                st.markdown("#### ☁️ Cloud (YOLOv8m) — *final result*")
                st.image(cloud_rgb, use_container_width=True)
                c_conf = result.get("confidence", 0)
                c_count = result.get("object_count", 0)
                c_lat = result.get("latency_ms", 0)
                c_classes = result.get("detected_classes", [])
                st.metric("Confidence", f"{c_conf:.3f}", delta=f"+{c_conf - e_conf:.3f} vs edge")
                st.metric("Objects", c_count, delta=f"+{c_count - e_count}" if c_count > e_count else str(c_count - e_count))
                st.metric("Latency", f"{c_lat:.1f} ms")
                st.metric("Upload", f"{len(image_bytes):,} bytes")
                if c_classes:
                    st.caption(f"Classes: {', '.join(c_classes)}")


# ---------------------------------------------------------------------------
# Page 2: Scenario Comparison
# ---------------------------------------------------------------------------


def page_comparison():
    st.header("📊 Scenario Comparison")

    dfs: dict[str, pd.DataFrame] = {}
    missing = []
    for key, path in SCENARIO_FILES.items():
        df = load_csv(path)
        if df is not None:
            dfs[key] = df
        else:
            missing.append(SCENARIO_LABELS[key])

    if missing:
        st.warning(
            f"No results found for: **{', '.join(missing)}**. "
            "Please run the experiment scripts first:\n"
            "```\npython edge/run_cloud_only.py\n"
            "python edge/run_edge_only.py\n"
            "python edge/run_edge_cloud.py\n```"
        )

    if not dfs:
        return

    # --- Summary table ---
    st.subheader("Summary Table")
    rows = []
    for key, df in dfs.items():
        numeric_df = df.copy()
        for col in ["latency_ms", "uploaded_bytes", "confidence", "cpu_percent", "ram_mb"]:
            if col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

        offload_ratio = (
            (numeric_df["offloaded"].astype(str).str.lower() == "true").sum()
            / len(numeric_df) * 100
            if "offloaded" in numeric_df.columns
            else 0.0
        )
        rows.append({
            "Scenario": SCENARIO_LABELS[key],
            "Avg Latency (ms)": f"{numeric_df['latency_ms'].mean():.2f}" if "latency_ms" in numeric_df else "—",
            "Total Upload (bytes)": f"{int(numeric_df['uploaded_bytes'].sum()):,}" if "uploaded_bytes" in numeric_df else "—",
            "Offload Ratio (%)": f"{offload_ratio:.1f}%" if key == "edge_cloud" else "N/A",
            "Avg Confidence": f"{numeric_df['confidence'].mean():.4f}" if "confidence" in numeric_df else "—",
            "Throughput (img/s)": f"{len(numeric_df) / max(numeric_df['latency_ms'].sum() / 1000, 0.001):.4f}" if "latency_ms" in numeric_df else "—",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Difficulty filter ---
    all_difficulties = set()
    for df in dfs.values():
        if "difficulty" in df.columns:
            all_difficulties.update(df["difficulty"].dropna().unique())
    difficulties = sorted(all_difficulties)

    selected_diff = st.multiselect(
        "Filter by difficulty",
        options=difficulties,
        default=difficulties,
    )

    filtered: dict[str, pd.DataFrame] = {}
    for key, df in dfs.items():
        if "difficulty" in df.columns and selected_diff:
            filtered[key] = df[df["difficulty"].isin(selected_diff)].copy()
        else:
            filtered[key] = df.copy()

    for key, df in filtered.items():
        for col in ["latency_ms", "uploaded_bytes", "confidence"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Chart 1: Latency ---
    st.subheader("📈 Chart 1: Average Latency (ms)")
    lat_data = [
        {"Scenario": SCENARIO_LABELS[k], "Avg Latency (ms)": df["latency_ms"].mean()}
        for k, df in filtered.items()
        if "latency_ms" in df.columns
    ]
    if lat_data:
        fig1 = px.bar(
            pd.DataFrame(lat_data),
            x="Scenario", y="Avg Latency (ms)",
            color="Scenario",
            color_discrete_map={SCENARIO_LABELS[k]: v for k, v in COLORS.items()},
            text_auto=".1f",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: Bandwidth ---
    st.subheader("📡 Chart 2: Total Upload Bandwidth (bytes)")
    bw_data = [
        {"Scenario": SCENARIO_LABELS[k], "Total Upload (bytes)": df["uploaded_bytes"].sum()}
        for k, df in filtered.items()
        if "uploaded_bytes" in df.columns
    ]
    if bw_data:
        fig2 = px.bar(
            pd.DataFrame(bw_data),
            x="Scenario", y="Total Upload (bytes)",
            color="Scenario",
            color_discrete_map={SCENARIO_LABELS[k]: v for k, v in COLORS.items()},
            text_auto=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Chart 3: Offload Ratio ---
    st.subheader("🔀 Chart 3: Offload Ratio (Edge-Cloud)")
    if "edge_cloud" in filtered and "offloaded" in filtered["edge_cloud"].columns:
        ec_df = filtered["edge_cloud"]
        offloaded_count = (ec_df["offloaded"].astype(str).str.lower() == "true").sum()
        edge_count = len(ec_df) - offloaded_count
        fig3 = px.pie(
            values=[edge_count, offloaded_count],
            names=["Processed at Edge", "Offloaded to Cloud"],
            color_discrete_sequence=["#00CC96", "#EF553B"],
            title=f"Offload ratio: {offloaded_count / max(len(ec_df), 1) * 100:.1f}%",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No edge-cloud data available to display offload ratio.")

    # --- Chart 4: Confidence ---
    st.subheader("🎯 Chart 4: Average Confidence")
    conf_data = [
        {"Scenario": SCENARIO_LABELS[k], "Avg Confidence": df["confidence"].mean()}
        for k, df in filtered.items()
        if "confidence" in df.columns
    ]
    if conf_data:
        fig4 = px.bar(
            pd.DataFrame(conf_data),
            x="Scenario", y="Avg Confidence",
            color="Scenario",
            color_discrete_map={SCENARIO_LABELS[k]: v for k, v in COLORS.items()},
            text_auto=".4f",
            range_y=[0, 1],
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Quality table by difficulty ---
    if difficulties:
        st.subheader("📋 Quality by Difficulty")
        quality_rows = []
        for diff in difficulties:
            for key, df in dfs.items():
                if "difficulty" not in df.columns:
                    continue
                sub = df[df["difficulty"] == diff].copy()
                if sub.empty:
                    continue
                for col in ["confidence", "object_count"]:
                    if col in sub.columns:
                        sub[col] = pd.to_numeric(sub[col], errors="coerce")
                quality_rows.append({
                    "Difficulty": diff,
                    "Scenario": SCENARIO_LABELS[key],
                    "Avg Confidence": f"{sub['confidence'].mean():.4f}" if "confidence" in sub else "—",
                    "Avg Object Count": f"{sub['object_count'].mean():.2f}" if "object_count" in sub else "—",
                    "Images": len(sub),
                })
        if quality_rows:
            st.dataframe(pd.DataFrame(quality_rows), use_container_width=True)

    # --- Improvement summary ---
    if "cloud_only" in dfs and "edge_cloud" in dfs:
        st.subheader("📉 Edge-Cloud vs Cloud-Only Improvement")
        co_bw = pd.to_numeric(dfs["cloud_only"]["uploaded_bytes"], errors="coerce").sum()
        ec_bw = pd.to_numeric(dfs["edge_cloud"]["uploaded_bytes"], errors="coerce").sum()
        if co_bw > 0:
            bw_improvement = (co_bw - ec_bw) / co_bw * 100
            st.metric(
                "Bandwidth savings",
                f"{bw_improvement:.1f}%",
                delta=f"-{co_bw - ec_bw:,} bytes",
            )

    if "edge_only" in dfs and "edge_cloud" in dfs:
        eo_hard = dfs["edge_only"]
        ec_hard = dfs["edge_cloud"]
        if "difficulty" in eo_hard.columns:
            eo_hard = eo_hard[eo_hard["difficulty"] == "hard"]
            ec_hard = ec_hard[ec_hard["difficulty"] == "hard"] if "difficulty" in ec_hard.columns else ec_hard
        eo_conf = pd.to_numeric(eo_hard["confidence"], errors="coerce").mean() if "confidence" in eo_hard.columns else 0
        ec_conf = pd.to_numeric(ec_hard["confidence"], errors="coerce").mean() if "confidence" in ec_hard.columns else 0
        if eo_conf > 0:
            conf_improvement = (ec_conf - eo_conf) / eo_conf * 100
            st.metric(
                "Confidence improvement (hard images)",
                f"{ec_conf:.4f}",
                delta=f"+{conf_improvement:.1f}% vs edge-only",
            )


# ---------------------------------------------------------------------------
# Page 3: Video Analysis
# ---------------------------------------------------------------------------


def page_video(threshold: float = CONFIDENCE_THRESHOLD, threshold_avg: float = 0.4, object_threshold: int = 10):
    st.header("🎬 Video Analysis")
    st.info(
        f"Upload a video file (MP4, AVI). Each frame will be processed using the "
        f"edge-cloud mechanism (confidence threshold: {CONFIDENCE_THRESHOLD})."
    )

    uploaded_video = st.file_uploader(
        "Upload video (MP4, AVI, max 100MB)",
        type=["mp4", "avi"],
    )

    if uploaded_video is None:
        return

    if uploaded_video.size > 100 * 1024 * 1024:
        st.error("Video file exceeds 100MB limit.")
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    st.write(f"**Total frames:** {total_frames} | **FPS:** {fps:.1f}")

    max_frames = st.slider("Max frames to process", 1, min(total_frames, 100), 20)
    process_btn = st.button("▶️ Start processing")

    if not process_btn:
        cap.release()
        os.unlink(video_path)
        return

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    frame_idx = 0
    processed = 0

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % max(1, total_frames // max_frames) != 0:
            continue

        _, buf = cv2.imencode(".jpg", frame)
        frame_bytes = buf.tobytes()

        edge_result = call_edge(frame_bytes)
        if edge_result:
            keep, _ = _should_keep_at_edge(edge_result, threshold, threshold_avg, object_threshold)
            if keep:
                result = edge_result
                place = "📱 Edge"
            else:
                result = call_cloud(frame_bytes, f"frame_{frame_idx:04d}.jpg")
                place = "☁️ Cloud" if result else "❌ Error"
        else:
            result = call_cloud(frame_bytes, f"frame_{frame_idx:04d}.jpg")
            place = "☁️ Cloud" if result else "❌ Error"

        if result:
            annotated = draw_bounding_boxes(frame, result.get("objects", []))
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
            info_placeholder.write(
                f"**Frame {frame_idx}** | Processed at: {place} | "
                f"Confidence: {result.get('confidence', 0):.3f} | "
                f"Latency: {result.get('latency_ms', 0):.1f} ms | "
                f"Objects: {result.get('object_count', 0)}"
            )

        processed += 1
        progress.progress(processed / max_frames)

    cap.release()
    os.unlink(video_path)
    st.success(f"Processed {processed} frames.")


# ---------------------------------------------------------------------------
# Page 4: Demo Cases — Prototype Evidence
# ---------------------------------------------------------------------------


def page_demo_cases(threshold: float = CONFIDENCE_THRESHOLD, threshold_avg: float = 0.4, object_threshold: int = 10):
    st.header("🧪 Demo Cases — Prototype Evidence")
    st.caption(
        "Upload images to demonstrate three key scenarios. "
        "Each case proves a different aspect of the edge-cloud system."
    )

    tab1, tab2, tab3 = st.tabs([
        "✅ Case 1: Keep at Edge",
        "☁️ Case 2: Offload to Cloud",
        "📊 Case 3: Scenario Comparison",
    ])

    # ── Case 1: Keep at Edge ────────────────────────────────────────────────
    with tab1:
        st.subheader("Case 1: Simple Image — Keep at Edge")
        st.markdown(
            "**Goal:** Show that a clear, simple image with few objects and high confidence "
            "is processed entirely at the edge — **zero bandwidth used**."
        )
        st.markdown(
            f"**Conditions to keep at edge:**  \n"
            f"- max_confidence ≥ **{threshold:.2f}**  \n"
            f"- avg_confidence ≥ **{threshold_avg:.2f}**  \n"
            f"- object_count ≤ **{object_threshold}**"
        )

        uploaded1 = st.file_uploader(
            "Upload a simple/clear image (few objects, good lighting)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="demo_case1",
        )
        if uploaded1:
            img_bytes = uploaded1.read()
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with st.spinner("Running edge inference..."):
                edge_result = call_edge(img_bytes, uploaded1.name)

            if edge_result and img_bgr is not None:
                keep, metrics = _should_keep_at_edge(edge_result, threshold, threshold_avg, object_threshold)
                m = metrics

                annotated = draw_bounding_boxes(img_bgr, edge_result.get("objects", []))
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Edge result", use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Max confidence", f"{m['max_conf']:.3f}")
                c2.metric("Avg confidence", f"{m['avg_conf']:.3f}")
                c3.metric("Objects", m['object_count'])
                c4.metric("Upload", "0 bytes ✅")

                if keep:
                    st.success(
                        f"✅ **Keep at Edge** — All conditions met!\n\n"
                        f"max_conf {m['max_conf']:.3f} ≥ {threshold:.2f} | "
                        f"avg_conf {m['avg_conf']:.3f} ≥ {threshold_avg:.2f} | "
                        f"objects {m['object_count']} ≤ {object_threshold}\n\n"
                        f"**Bandwidth saved: {len(img_bytes):,} bytes (100% savings vs cloud-only)**"
                    )
                else:
                    failed = []
                    if not m['cond1']: failed.append(f"max_conf {m['max_conf']:.3f} < {threshold:.2f}")
                    if not m['cond2']: failed.append(f"avg_conf {m['avg_conf']:.3f} < {threshold_avg:.2f}")
                    if not m['cond3']: failed.append(f"objects {m['object_count']} > {object_threshold}")
                    st.warning(
                        f"⚠️ This image would be offloaded. Failed: {' | '.join(failed)}\n\n"
                        "Try an image with fewer, clearer objects, or adjust thresholds in sidebar."
                    )

    # ── Case 2: Offload to Cloud ─────────────────────────────────────────────
    with tab2:
        st.subheader("Case 2: Complex Image — Offload to Cloud")
        st.markdown(
            "**Goal:** Show that a complex image (many objects, low confidence, or crowded scene) "
            "triggers offload to cloud — **cloud model provides better accuracy**."
        )

        uploaded2 = st.file_uploader(
            "Upload a complex image (many objects, crowded, or low light)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="demo_case2",
        )
        if uploaded2:
            img_bytes = uploaded2.read()
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with st.spinner("Running edge + cloud inference..."):
                edge_result = call_edge(img_bytes, uploaded2.name)
                cloud_result = call_cloud(img_bytes, uploaded2.name)

            if edge_result and cloud_result and img_bgr is not None:
                keep, metrics = _should_keep_at_edge(edge_result, threshold, threshold_avg, object_threshold)
                m = metrics

                col_e, col_c = st.columns(2)
                with col_e:
                    st.markdown("#### 📱 Edge (YOLOv8n)")
                    ann_e = draw_bounding_boxes(img_bgr, edge_result.get("objects", []))
                    st.image(cv2.cvtColor(ann_e, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.metric("Max confidence", f"{m['max_conf']:.3f}")
                    st.metric("Avg confidence", f"{m['avg_conf']:.3f}")
                    st.metric("Objects", m['object_count'])
                    st.metric("Upload", "0 bytes")

                with col_c:
                    st.markdown("#### ☁️ Cloud (YOLOv8m)")
                    ann_c = draw_bounding_boxes(img_bgr, cloud_result.get("objects", []))
                    st.image(cv2.cvtColor(ann_c, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.metric("Max confidence", f"{cloud_result.get('confidence', 0):.3f}",
                              delta=f"+{cloud_result.get('confidence', 0) - m['max_conf']:.3f}")
                    st.metric("Objects", cloud_result.get("object_count", 0),
                              delta=f"+{cloud_result.get('object_count', 0) - m['object_count']}")
                    st.metric("Upload", f"{len(img_bytes):,} bytes")

                if not keep:
                    failed = []
                    if not m['cond1']: failed.append(f"max_conf {m['max_conf']:.3f} < {threshold:.2f}")
                    if not m['cond2']: failed.append(f"avg_conf {m['avg_conf']:.3f} < {threshold_avg:.2f}")
                    if not m['cond3']: failed.append(f"objects {m['object_count']} > {object_threshold}")
                    st.warning(
                        f"☁️ **Offload to Cloud** — Condition(s) failed: {' | '.join(failed)}\n\n"
                        f"Cloud improved confidence: {m['max_conf']:.3f} → {cloud_result.get('confidence', 0):.3f}"
                    )
                else:
                    st.info("This image would be kept at edge with current thresholds. Try adjusting thresholds in sidebar to force offload.")

    # ── Case 3: Scenario Comparison ──────────────────────────────────────────
    with tab3:
        st.subheader("Case 3: Same Image — Three Scenarios Compared")
        st.markdown(
            "**Goal:** Run the same image through all three scenarios and compare "
            "latency, bandwidth, and confidence side by side."
        )

        uploaded3 = st.file_uploader(
            "Upload any image to compare all three scenarios",
            type=["jpg", "jpeg", "png", "bmp"],
            key="demo_case3",
        )
        if uploaded3:
            img_bytes = uploaded3.read()
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with st.spinner("Running all three scenarios..."):
                edge_result = call_edge(img_bytes, uploaded3.name)
                cloud_result = call_cloud(img_bytes, uploaded3.name)

            if edge_result and cloud_result and img_bgr is not None:
                keep, metrics = _should_keep_at_edge(edge_result, threshold, threshold_avg, object_threshold)
                m = metrics

                # Edge-cloud final result
                ec_result = edge_result if keep else cloud_result
                ec_place = "Edge" if keep else "Cloud"
                ec_upload = 0 if keep else len(img_bytes)

                # Build comparison table
                rows = [
                    {
                        "Scenario": "Cloud-Only",
                        "Processed at": "☁️ Cloud",
                        "Confidence": f"{cloud_result.get('confidence', 0):.3f}",
                        "Objects": cloud_result.get("object_count", 0),
                        "Latency (ms)": f"{cloud_result.get('latency_ms', 0):.1f}",
                        "Upload (bytes)": f"{len(img_bytes):,}",
                    },
                    {
                        "Scenario": "Edge-Only",
                        "Processed at": "📱 Edge",
                        "Confidence": f"{m['max_conf']:.3f}",
                        "Objects": m['object_count'],
                        "Latency (ms)": f"{edge_result.get('latency_ms', 0):.1f}",
                        "Upload (bytes)": "0",
                    },
                    {
                        "Scenario": "Edge-Cloud",
                        "Processed at": f"{'📱 Edge' if keep else '☁️ Cloud'}",
                        "Confidence": f"{ec_result.get('confidence', 0):.3f}",
                        "Objects": ec_result.get("object_count", 0),
                        "Latency (ms)": f"{ec_result.get('latency_ms', 0):.1f}",
                        "Upload (bytes)": f"{ec_upload:,}",
                    },
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Visual comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Cloud-Only**")
                    ann = draw_bounding_boxes(img_bgr, cloud_result.get("objects", []))
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown("**Edge-Only**")
                    ann = draw_bounding_boxes(img_bgr, edge_result.get("objects", []))
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col3:
                    st.markdown(f"**Edge-Cloud** → {ec_place}")
                    ann = draw_bounding_boxes(img_bgr, ec_result.get("objects", []))
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

                # Key insight
                bw_savings = (1 - ec_upload / len(img_bytes)) * 100 if len(img_bytes) > 0 else 0
                st.info(
                    f"**Key insight:** Edge-Cloud processed at **{ec_place}** with "
                    f"confidence **{ec_result.get('confidence', 0):.3f}** and "
                    f"**{bw_savings:.0f}% bandwidth savings** vs Cloud-Only."
                )


# ---------------------------------------------------------------------------
# Page 5: Experiment Results — Table 5.3
# ---------------------------------------------------------------------------


def _compute_p95(values: list[float]) -> float:
    """Compute 95th percentile."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def page_experiment_results():
    st.header("📋 Experiment Results — Table 5.3")
    st.caption(
        "Automatically computed from CSV result files. "
        "Run the experiment scripts first to populate the data."
    )

    # Load all three scenario CSVs
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in SCENARIO_FILES.items():
        df = load_csv(path)
        if df is not None:
            for col in ["latency_ms", "uploaded_bytes", "confidence",
                        "object_count", "cpu_percent", "ram_mb"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            dfs[key] = df

    if not dfs:
        st.warning(
            "No experiment data found. Please run the experiment scripts first:\n"
            "```\npython edge/run_cloud_only.py\n"
            "python edge/run_edge_only.py\n"
            "python edge/run_edge_cloud.py\n```"
        )
        return

    def _get(key: str, col: str, agg: str = "mean") -> str:
        if key not in dfs or col not in dfs[key].columns:
            return "—"
        vals = dfs[key][col].dropna().tolist()
        if not vals:
            return "—"
        if agg == "mean":
            return f"{sum(vals) / len(vals):.2f}"
        elif agg == "sum":
            return f"{int(sum(vals)):,}"
        elif agg == "p95":
            return f"{_compute_p95(vals):.2f}"
        elif agg == "throughput":
            total_ms = dfs[key]["latency_ms"].dropna().sum()
            n = len(dfs[key])
            return f"{n / max(total_ms / 1000, 0.001):.4f}" if total_ms > 0 else "—"
        return "—"

    def _offload_ratio(key: str) -> str:
        if key not in dfs or "offloaded" not in dfs[key].columns:
            return "N/A"
        df = dfs[key]
        total = len(df)
        offloaded = (df["offloaded"].astype(str).str.lower() == "true").sum()
        return f"{offloaded / total * 100:.1f}%" if total > 0 else "—"

    # ── Table 5.3 ────────────────────────────────────────────────────────────
    st.subheader("Table 5.3: Experiment Results Summary")

    metrics = [
        ("Avg Latency (ms)",        "latency_ms",      "mean"),
        ("P95 Latency (ms)",        "latency_ms",      "p95"),
        ("Total Upload (bytes)",    "uploaded_bytes",  "sum"),
        ("Throughput (img/s)",      "latency_ms",      "throughput"),
        ("Avg Confidence",          "confidence",      "mean"),
        ("Avg Object Count",        "object_count",    "mean"),
        ("Avg CPU Usage (%)",       "cpu_percent",     "mean"),
        ("Avg RAM Usage (MB)",      "ram_mb",          "mean"),
    ]

    rows = []
    for label, col, agg in metrics:
        rows.append({
            "Metric": label,
            "Cloud-Only": _get("cloud_only", col, agg),
            "Edge-Only":  _get("edge_only",  col, agg),
            "Edge-Cloud": _get("edge_cloud", col, agg),
        })

    # Add offload ratio row
    rows.append({
        "Metric": "Offload Ratio (%)",
        "Cloud-Only": "100%",
        "Edge-Only":  "0%",
        "Edge-Cloud": _offload_ratio("edge_cloud"),
    })

    # Add bandwidth savings row
    co_bw = dfs["cloud_only"]["uploaded_bytes"].sum() if "cloud_only" in dfs else 0
    ec_bw = dfs["edge_cloud"]["uploaded_bytes"].sum() if "edge_cloud" in dfs else 0
    bw_savings = f"{(co_bw - ec_bw) / co_bw * 100:.1f}%" if co_bw > 0 else "—"
    rows.append({
        "Metric": "Bandwidth Savings vs Cloud-Only",
        "Cloud-Only": "0%",
        "Edge-Only":  "100%",
        "Edge-Cloud": bw_savings,
    })

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ── Download button ───────────────────────────────────────────────────────
    csv_export = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Table as CSV",
        data=csv_export,
        file_name="experiment_results_table53.csv",
        mime="text/csv",
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Visual Summary")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Latency comparison
        lat_rows = []
        for key in ["cloud_only", "edge_only", "edge_cloud"]:
            if key in dfs and "latency_ms" in dfs[key].columns:
                lat_rows.append({
                    "Scenario": SCENARIO_LABELS[key],
                    "Avg Latency (ms)": dfs[key]["latency_ms"].mean(),
                    "P95 Latency (ms)": _compute_p95(dfs[key]["latency_ms"].dropna().tolist()),
                })
        if lat_rows:
            lat_df = pd.DataFrame(lat_rows)
            fig_lat = px.bar(
                lat_df.melt(id_vars="Scenario", var_name="Metric", value_name="ms"),
                x="Scenario", y="ms", color="Metric", barmode="group",
                title="Latency Comparison (ms)",
                color_discrete_sequence=["#636EFA", "#EF553B"],
            )
            st.plotly_chart(fig_lat, use_container_width=True)

    with chart_col2:
        # Bandwidth comparison
        bw_rows = []
        for key in ["cloud_only", "edge_only", "edge_cloud"]:
            if key in dfs and "uploaded_bytes" in dfs[key].columns:
                bw_rows.append({
                    "Scenario": SCENARIO_LABELS[key],
                    "Total Upload (bytes)": int(dfs[key]["uploaded_bytes"].sum()),
                })
        if bw_rows:
            fig_bw = px.bar(
                pd.DataFrame(bw_rows),
                x="Scenario", y="Total Upload (bytes)",
                color="Scenario",
                color_discrete_map={SCENARIO_LABELS[k]: v for k, v in COLORS.items()},
                title="Total Upload Bandwidth (bytes)",
                text_auto=True,
            )
            st.plotly_chart(fig_bw, use_container_width=True)

    # Confidence comparison
    conf_rows = []
    for key in ["cloud_only", "edge_only", "edge_cloud"]:
        if key in dfs and "confidence" in dfs[key].columns:
            conf_rows.append({
                "Scenario": SCENARIO_LABELS[key],
                "Avg Confidence": dfs[key]["confidence"].mean(),
            })
    if conf_rows:
        fig_conf = px.bar(
            pd.DataFrame(conf_rows),
            x="Scenario", y="Avg Confidence",
            color="Scenario",
            color_discrete_map={SCENARIO_LABELS[k]: v for k, v in COLORS.items()},
            title="Average Confidence Score",
            text_auto=".4f",
            range_y=[0, 1],
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Per-image raw data ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Raw Data per Image")
    selected_scenario = st.selectbox(
        "Select scenario to inspect",
        options=list(dfs.keys()),
        format_func=lambda x: SCENARIO_LABELS.get(x, x),
    )
    if selected_scenario in dfs:
        st.dataframe(dfs[selected_scenario], use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Edge-Cloud Demo Dashboard",
        page_icon="🌐",
        layout="wide",
    )

    st.title("🌐 Edge-Cloud Computing Demo")
    st.caption("Multimedia transmission and large-scale image semantic analysis via Edge-Cloud architecture")

    # ── Sidebar navigation ──────────────────────────────────────────────────
    page = st.sidebar.radio(
        "Navigation",
        options=["single_image", "comparison", "demo_cases", "experiment_results", "video"],
        format_func=lambda x: {
            "single_image": "🖼️ Single Image Analysis",
            "comparison": "📊 Scenario Comparison",
            "demo_cases": "🧪 Demo Cases",
            "experiment_results": "📋 Experiment Results",
            "video": "🎬 Video Analysis",
        }[x],
    )

    st.sidebar.markdown("---")

    # ── Dynamic confidence threshold ────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Edge-Cloud Settings")

    if "threshold" not in st.session_state:
        st.session_state.threshold = CONFIDENCE_THRESHOLD
    if "threshold_avg" not in st.session_state:
        st.session_state.threshold_avg = 0.4
    if "object_threshold" not in st.session_state:
        st.session_state.object_threshold = 10

    st.session_state.threshold = st.sidebar.slider(
        "Max Confidence Threshold",
        min_value=0.01, max_value=1.0,
        value=st.session_state.threshold,
        step=0.01,
        help="Keep at Edge if max_confidence ≥ this value",
    )
    threshold = st.session_state.threshold

    st.session_state.threshold_avg = st.sidebar.slider(
        "Avg Confidence Threshold",
        min_value=0.01, max_value=1.0,
        value=st.session_state.threshold_avg,
        step=0.01,
        help="Keep at Edge if average_confidence ≥ this value",
    )
    threshold_avg = st.session_state.threshold_avg

    st.session_state.object_threshold = st.sidebar.slider(
        "Max Object Count",
        min_value=1, max_value=50,
        value=st.session_state.object_threshold,
        step=1,
        help="Keep at Edge if object_count ≤ this value (too many objects → offload)",
    )
    object_threshold = st.session_state.object_threshold

    st.sidebar.markdown(
        f"**Keep at Edge if:**  \n"
        f"`max_conf ≥ {threshold:.2f}`  \n"
        f"`avg_conf ≥ {threshold_avg:.2f}`  \n"
        f"`objects ≤ {object_threshold}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Cloud URL:** `{CLOUD_URL}`")
    st.sidebar.markdown(f"**Edge URL:** `{EDGE_URL}`")

    # ── Dataset management ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Dataset")
    dataset_dir = os.environ.get("DATASET_DIR", "dataset")
    img_count = sum(
        len([f for f in os.listdir(os.path.join(dataset_dir, "images", d))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        for d in ("easy", "medium", "hard")
        if os.path.isdir(os.path.join(dataset_dir, "images", d))
    ) if os.path.isdir(os.path.join(dataset_dir, "images")) else 0

    if img_count > 0:
        st.sidebar.success(f"✅ {img_count} images available")
    else:
        st.sidebar.warning("⚠️ No images found")

    if st.sidebar.button("⬇️ Download Test Dataset", help="Download 30 sample images (10 per difficulty level)"):
        with st.sidebar.status("Downloading...", expanded=True) as status:
            try:
                import urllib.request
                SAMPLE_URLS = {
                    "easy":   [(f"https://picsum.photos/id/{200+i}/640/480", f"easy_{i+1:03d}.jpg") for i in range(10)],
                    "medium": [(f"https://picsum.photos/id/{400+i}/640/480", f"medium_{i+1:03d}.jpg") for i in range(10)],
                    "hard":   [(f"https://picsum.photos/id/{500+i}/640/480", f"hard_{i+1:03d}.jpg") for i in range(10)],
                }
                downloaded = 0
                for difficulty, images in SAMPLE_URLS.items():
                    out_dir = os.path.join(dataset_dir, "images", difficulty)
                    os.makedirs(out_dir, exist_ok=True)
                    for url, fname in images:
                        out_path = os.path.join(out_dir, fname)
                        if not os.path.exists(out_path):
                            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=15) as resp:
                                with open(out_path, "wb") as f:
                                    f.write(resp.read())
                        downloaded += 1
                        st.write(f"✓ {difficulty}/{fname}")
                status.update(label=f"✅ Downloaded {downloaded} images", state="complete")
            except Exception as exc:
                status.update(label=f"❌ Error: {exc}", state="error")

    if page == "single_image":
        page_single_image(threshold, threshold_avg, object_threshold)
    elif page == "comparison":
        page_comparison()
    elif page == "demo_cases":
        page_demo_cases(threshold, threshold_avg, object_threshold)
    elif page == "experiment_results":
        page_experiment_results()
    else:
        page_video(threshold, threshold_avg, object_threshold)


if __name__ == "__main__":
    main()
