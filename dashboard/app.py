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


def _compute_offload_confidence(result: dict, strategy: str) -> float:
    """Compute the confidence value used for offload decision based on strategy."""
    objects = result.get("objects", [])
    if not objects:
        return 0.0
    confs = [o.get("confidence", 0.0) for o in objects]
    if strategy == "max":
        return max(confs)
    elif strategy == "avg":
        return sum(confs) / len(confs)
    else:  # min
        return min(confs)


def page_single_image(threshold: float = CONFIDENCE_THRESHOLD, offload_strategy: str = "max"):
    st.header("🖼️ Single Image Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload image (JPEG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )
    with col2:
        scenario = st.selectbox(
            "Select scenario",
            options=["edge_cloud", "cloud_only", "edge_only"],
            format_func=lambda x: SCENARIO_LABELS[x],
        )
        if scenario == "edge_cloud":
            st.info(f"Current threshold: **{threshold:.2f}**\n\nAdjust in sidebar ←")

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
    edge_conf_used = None  # confidence value used for offload decision

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
                edge_conf_used = _compute_offload_confidence(edge_result, offload_strategy)
                strategy_label = {"max": "max", "avg": "avg", "min": "min"}[offload_strategy]
                if edge_conf_used >= threshold:
                    result = edge_result
                    processing_place = "📱 Edge"
                    st.info(f"{strategy_label} confidence {edge_conf_used:.3f} ≥ threshold {threshold:.2f} → processed at **Edge**")
                else:
                    st.warning(f"{strategy_label} confidence {edge_conf_used:.3f} < threshold {threshold:.2f} → offloading to **Cloud**")
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

    st.image(annotated_rgb, caption="Detection result", use_column_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed at", processing_place)
    m2.metric("Confidence", f"{result.get('confidence', 0):.3f}")
    m3.metric("Latency", f"{result.get('latency_ms', 0):.1f} ms")
    m4.metric("Objects detected", result.get("object_count", 0))

    if result.get("detected_classes"):
        st.write("**Detected classes:**", ", ".join(result["detected_classes"]))

    # Show edge-cloud decision explanation
    if scenario == "edge_cloud" and edge_conf_used is not None:
        final_conf = result.get("confidence", 0) if result else 0
        st.markdown("---")
        st.markdown("**Edge-Cloud Decision Logic:**")
        col_a, col_b, col_c = st.columns(3)
        strategy_labels = {"max": "Edge max conf", "avg": "Edge avg conf", "min": "Edge min conf"}
        col_a.metric(strategy_labels[offload_strategy], f"{edge_conf_used:.3f}")
        col_b.metric("Threshold", f"{threshold:.2f}")
        if processing_place.startswith("📱"):
            col_c.metric("Decision", "✅ Keep at Edge", delta="No upload (0 bytes)")
            st.success(f"Edge confidence {edge_conf_used:.3f} ≥ threshold {threshold:.2f} → Edge result accepted. Bandwidth saved.")
        else:
            col_c.metric("Decision", "☁️ Offload to Cloud", delta=f"+{len(image_bytes):,} bytes uploaded")
            st.warning(
                f"Edge confidence {edge_conf_used:.3f} < threshold {threshold:.2f} → Offloaded to Cloud.\n\n"
                f"Cloud result: confidence **{final_conf:.3f}** ({'+' if final_conf > edge_conf_used else ''}{final_conf - edge_conf_used:.3f} vs edge)"
            )


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


def page_video(threshold: float = CONFIDENCE_THRESHOLD, offload_strategy: str = "max"):
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
        if edge_result and edge_result["confidence"] >= threshold:
            result = edge_result
            place = "📱 Edge"
        else:
            result = call_cloud(frame_bytes, f"frame_{frame_idx:04d}.jpg")
            place = "☁️ Cloud" if result else "❌ Error"

        if result:
            annotated = draw_bounding_boxes(frame, result.get("objects", []))
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, caption=f"Frame {frame_idx}", use_column_width=True)
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
        options=["single_image", "comparison", "video"],
        format_func=lambda x: {
            "single_image": "🖼️ Single Image Analysis",
            "comparison": "📊 Scenario Comparison",
            "video": "🎬 Video Analysis",
        }[x],
    )

    st.sidebar.markdown("---")

    # ── Dynamic confidence threshold ────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Edge-Cloud Settings")
    if "threshold" not in st.session_state:
        st.session_state.threshold = CONFIDENCE_THRESHOLD

    st.session_state.threshold = st.sidebar.slider(
        "Offload Threshold",
        min_value=0.10,
        max_value=0.99,
        value=st.session_state.threshold,
        step=0.05,
        help=(
            "Images with edge confidence BELOW this value are offloaded to Cloud.\n\n"
            "↑ Higher → more offloading → better accuracy, more bandwidth\n"
            "↓ Lower  → less offloading → faster, less bandwidth"
        ),
    )
    threshold = st.session_state.threshold

    # Offload strategy
    if "offload_strategy" not in st.session_state:
        st.session_state.offload_strategy = "max"

    st.session_state.offload_strategy = st.sidebar.selectbox(
        "Offload Strategy",
        options=["max", "avg", "min"],
        index=["max", "avg", "min"].index(st.session_state.offload_strategy),
        format_func=lambda x: {
            "max": "Max confidence (default)",
            "avg": "Average confidence",
            "min": "Min confidence (strictest)",
        }[x],
        help=(
            "Which confidence value to compare against threshold:\n\n"
            "• Max — offload only if best detection is uncertain\n"
            "• Avg — offload if average detection quality is low\n"
            "• Min — offload if ANY detection is uncertain (strictest)"
        ),
    )
    offload_strategy = st.session_state.offload_strategy

    # Show threshold impact hint
    if threshold <= 0.3:
        st.sidebar.caption("🟢 Very low — almost everything stays at edge")
    elif threshold <= 0.6:
        st.sidebar.caption("🟡 Balanced — default research setting")
    elif threshold <= 0.8:
        st.sidebar.caption("🟠 High — more images offloaded to cloud")
    else:
        st.sidebar.caption("🔴 Very high — most images offloaded to cloud")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Cloud URL:** `{CLOUD_URL}`")
    st.sidebar.markdown(f"**Edge URL:** `{EDGE_URL}`")

    if page == "single_image":
        page_single_image(threshold, offload_strategy)
    elif page == "comparison":
        page_comparison()
    else:
        page_video(threshold, offload_strategy)


if __name__ == "__main__":
    main()
