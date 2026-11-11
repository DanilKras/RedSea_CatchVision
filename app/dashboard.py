from __future__ import annotations

import base64
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5000").rstrip("/")

DEFAULT_SCENARIOS = [
    "clean",
    "sun_glare",
    "water_droplets",
    "low_light",
    "motion_blur",
    "marker_occlusion_1",
    "marker_occlusion_2",
]

st.set_page_config(
    page_title="RedSea-CatchVision",
    page_icon="🐟",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def decode_b64_image(b64_str: str) -> bytes:
    return base64.b64decode(b64_str.encode("ascii"))


@st.cache_data(ttl=15)
def check_api_health(base_url: str) -> dict[str, object] | None:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=44, b=16),
    )
    return fig


st.title("RedSea-CatchVision")
st.caption(
    "Automated reef fishery monitoring: GroundingDINO + SAM 2.1 segmentation, "
    "ArUco metric sizing, and open-set BioCLIP identification."
)

with st.sidebar:
    st.subheader("System Status")
    health = check_api_health(API_BASE_URL)
    if health:
        st.success(f"Inference API online ({health.get('device', 'cpu')})")
    else:
        st.error(f"Inference API offline ({API_BASE_URL})")

    c_doc, c_ml = st.columns(2)
    with c_doc:
        st.link_button("API Docs", f"{PUBLIC_API_URL}/docs", use_container_width=True)
    with c_ml:
        st.link_button("MLflow", MLFLOW_UI_URL, use_container_width=True)

    st.divider()
    st.subheader("Survey & Gating")
    site_name = st.text_input("Landing Site", value="Thuwal_KAUST_Dock")
    box_thresh = st.slider("Detection threshold", 0.15, 0.60, 0.28, 0.02)
    min_conf_auto = st.slider("Min softmax confidence", 0.40, 0.95, 0.70, 0.05)
    min_cos_auto = st.slider("Min raw cosine (OOD gate)", 0.10, 0.50, 0.22, 0.01)

    with st.expander("Advanced & OOD Demo", expanded=False):
        include_jewel_demo = st.checkbox(
            "Add non-Red-Sea control species (Hemichromis guttatus)",
            value=False,
        )

tab_inspect, tab_ledger, tab_bench = st.tabs(
    ["Dock Inspection", "Catch Ledger", "Field Stress Benchmark"]
)

with tab_inspect:
    col_up, col_act = st.columns([3, 1], vertical_alignment="bottom")
    with col_up:
        uploaded = st.file_uploader(
            "Upload catch photo (with or without ArUco mat)",
            type=["jpg", "jpeg", "png"],
        )
    with col_act:
        use_sample = st.button("Run Sample Frame", use_container_width=True)
        run_uploaded = st.button(
            "Inspect Photo",
            type="primary",
            disabled=uploaded is None,
            use_container_width=True,
        )

    if run_uploaded or use_sample:
        form_data = {
            "use_sample": str(bool(use_sample)).lower(),
            "site_name": site_name,
            "box_threshold": str(box_thresh),
            "min_conf_auto": str(min_conf_auto),
            "min_cos_auto": str(min_cos_auto),
            "include_jewel_demo": str(include_jewel_demo).lower(),
            "save_db": "false",
        }
        files = None
        if run_uploaded and uploaded is not None:
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "image/jpeg",
                )
            }

        with st.spinner("Running segmentation, homography rectification, and BioCLIP..."):
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/api/v1/inspect",
                    data=form_data,
                    files=files,
                    timeout=120.0,
                )
                resp.raise_for_status()
                st.session_state["last_inspection"] = resp.json()
            except Exception as exc:
                st.error(f"Inference API request failed: {exc}")

    result = st.session_state.get("last_inspection")
    if result is not None:
        rows = result.get("rows", [])
        n_fish = len(rows)
        n_review = sum(1 for r in rows if r.get("needs_review"))
        total_w = sum(r["weight_g"] for r in rows if r.get("weight_g") is not None)
        reproj = result.get("reproj_error_mm")
        calib_label = (
            f"{reproj:.3f} mm" if result.get("calibrated") and reproj is not None else "Skipped"
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Detected Specimens", n_fish)
        k2.metric("Reprojection Error", calib_label)
        k3.metric("Flagged for Review", n_review)
        k4.metric("Estimated Biomass (g)", f"{total_w:.1f}" if total_w > 0 else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Detection & SAM 2.1 Mask**")
            if result.get("overlay_b64"):
                st.image(decode_b64_image(result["overlay_b64"]), use_container_width=True)
        with c2:
            st.markdown("**Metric Rectification & Total Length**")
            if result.get("rectified_b64"):
                st.image(decode_b64_image(result["rectified_b64"]), use_container_width=True)
            else:
                st.info("No ArUco calibration mat detected — running uncalibrated species ID.")

        if rows:
            st.markdown("**Specimen Measurements & Open-Set Species ID**")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fish_id": st.column_config.NumberColumn("ID", format="%d"),
                    "species": "Scientific Name",
                    "common_name": "Common Name",
                    "local_ar": "Local Name",
                    "cls_conf": st.column_config.ProgressColumn(
                        "Softmax Conf", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "raw_cos": st.column_config.NumberColumn("Raw Cosine", format="%.3f"),
                    "margin": st.column_config.NumberColumn("Margin", format="%.2f"),
                    "needs_review": "Review",
                    "tl_cm": st.column_config.NumberColumn("TL (cm)", format="%.2f"),
                    "depth_cm": st.column_config.NumberColumn("Depth (cm)", format="%.2f"),
                    "weight_g": st.column_config.NumberColumn("Weight (g)", format="%.1f"),
                    "undersized": "Undersized",
                },
            )

            if st.button("Save Frame to Catch Ledger", type="primary"):
                insp_id = result["inspection_id"]
                try:
                    commit_resp = httpx.post(
                        f"{API_BASE_URL}/api/v1/inspect/{insp_id}/commit",
                        timeout=15.0,
                    )
                    commit_resp.raise_for_status()
                    payload = commit_resp.json()
                    st.success(
                        f"Saved frame `{payload['frame_id']}` "
                        f"({payload['fish_count']} specimens) to DuckDB."
                    )
                except Exception as exc:
                    st.error(f"Failed to commit frame: {exc}")
        else:
            st.warning("No fish detected above the active confidence threshold.")

with tab_ledger:
    try:
        ledger_resp = httpx.get(
            f"{API_BASE_URL}/api/v1/ledger",
            params={"limit": 100},
            timeout=10.0,
        )
        ledger_resp.raise_for_status()
        ledger = ledger_resp.json()
        sp_df = pd.DataFrame(ledger["species_summary"])
        recent_df = pd.DataFrame(ledger["recent_catches"])
    except Exception as exc:
        sp_df = pd.DataFrame()
        recent_df = pd.DataFrame()
        st.error(f"Could not fetch catch ledger: {exc}")

    if sp_df.empty:
        st.info("No catch records logged yet. Run an inspection and save it to populate DuckDB.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Logged Specimens", int(sp_df["fish_count"].sum()))
        m2.metric("Species Richness", len(sp_df))
        m3.metric("Total Biomass (kg)", f"{sp_df['total_biomass_kg'].sum():.3f}")
        total_fish = sp_df["fish_count"].sum()
        if total_fish > 0 and "pct_undersized" in sp_df:
            undersized_pct = float((sp_df["pct_undersized"] * sp_df["fish_count"]).sum() / total_fish)
        else:
            undersized_pct = 0.0
        m4.metric("Undersized Share", f"{undersized_pct:.1f}%")

        st.markdown("**Species Composition & Biomass**")
        st.dataframe(sp_df, use_container_width=True, hide_index=True)

        g1, g2 = st.columns(2)
        with g1:
            fig_sp = px.bar(
                sp_df,
                x="top_species",
                y="total_biomass_kg",
                color="pct_undersized",
                labels={
                    "top_species": "Species",
                    "total_biomass_kg": "Biomass (kg)",
                    "pct_undersized": "Undersized %",
                },
                title="Biomass & Undersized Ratio by Species",
            )
            st.plotly_chart(style_plotly(fig_sp), use_container_width=True)
        with g2:
            if not recent_df.empty and "tl_cm" in recent_df and recent_df["tl_cm"].notna().any():
                fig_len = px.histogram(
                    recent_df.dropna(subset=["tl_cm"]),
                    x="tl_cm",
                    color="top_species",
                    nbins=15,
                    labels={"tl_cm": "Total Length (cm)", "top_species": "Species"},
                    title="Total Length Distribution",
                )
                st.plotly_chart(style_plotly(fig_len), use_container_width=True)

        st.markdown("**Recent Catch Log**")
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

with tab_bench:
    with st.expander("Benchmark Configuration", expanded=True):
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            bench_img_str = st.text_input(
                "Reference image path (API host)",
                value="data/processed/test_run/synthetic_tilted.jpg",
            )
        with col_b:
            gt_len = st.number_input(
                "Ground-truth TL (mm)",
                min_value=10.0,
                max_value=2000.0,
                value=150.0,
            )
        with col_c:
            sev_choice = st.multiselect(
                "Severities",
                [0.2, 0.4, 0.6, 0.8],
                default=[0.4, 0.8],
            )

        selected_scenarios = st.multiselect(
            "Perturbation scenarios",
            DEFAULT_SCENARIOS,
            default=DEFAULT_SCENARIOS,
        )

        if st.button("Run Stress Benchmark", type="primary"):
            with st.spinner("Running perturbation suite on Inference API..."):
                try:
                    b_resp = httpx.post(
                        f"{API_BASE_URL}/api/v1/benchmark",
                        json={
                            "image_path": bench_img_str,
                            "ground_truth_length_mm": gt_len,
                            "scenarios": selected_scenarios,
                            "severities": sev_choice or [0.4],
                        },
                        timeout=300.0,
                    )
                    b_resp.raise_for_status()
                    st.success("Stress benchmark completed and logged to DuckDB + MLflow.")
                except Exception as exc:
                    st.error(f"Benchmark failed: {exc}")

    try:
        sum_resp = httpx.get(f"{API_BASE_URL}/api/v1/benchmark", timeout=10.0)
        sum_resp.raise_for_status()
        bench_payload = sum_resp.json()
        bench_summary = pd.DataFrame(bench_payload["summary"])
        sample_paths = bench_payload["sample_images"]
    except Exception as exc:
        bench_summary = pd.DataFrame()
        sample_paths = []
        st.error(f"Could not fetch benchmark summary: {exc}")

    if not bench_summary.empty:
        st.markdown("**Robustness Metrics Across Field Perturbations**")
        st.dataframe(bench_summary, use_container_width=True, hide_index=True)

        c_l, c_r = st.columns(2)
        with c_l:
            fig_err = px.bar(
                bench_summary,
                x="scenario",
                y="mape_length_pct",
                color="severity",
                barmode="group",
                labels={"scenario": "Scenario", "mape_length_pct": "Length Error (%)"},
                title="Metric Sizing Error (MAPE %)",
            )
            fig_err.add_hline(y=2.0, line_dash="dash", line_color="red")
            st.plotly_chart(style_plotly(fig_err), use_container_width=True)
        with c_r:
            fig_conf = px.bar(
                bench_summary,
                x="scenario",
                y="mean_cls_conf",
                color="severity",
                barmode="group",
                labels={"scenario": "Scenario", "mean_cls_conf": "Mean Confidence"},
                title="BioCLIP Top-1 Confidence",
            )
            st.plotly_chart(style_plotly(fig_conf), use_container_width=True)

    if sample_paths:
        st.markdown("**Perturbed Sample Gallery**")
        cols = st.columns(3)
        for idx, rel in enumerate(sample_paths):
            with cols[idx % 3]:
                try:
                    img_resp = httpx.get(
                        f"{API_BASE_URL}/api/v1/artifacts/{rel}",
                        timeout=5.0,
                    )
                    img_resp.raise_for_status()
                    st.image(
                        img_resp.content,
                        caption=rel.split("/")[-1],
                        use_container_width=True,
                    )
                except Exception:
                    pass
