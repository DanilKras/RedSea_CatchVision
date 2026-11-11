# RedSea-CatchVision

**Open-set reef fishery monitoring: zero-shot detection & segmentation, ArUco metric sizing, LWR biomass estimation, and dockside stress benchmarking.**

RedSea-CatchVision is an end-to-end computer vision and MLOps pipeline designed for artisanal reef fishery landing surveys in the Red Sea. It combines foundation vision models with perspective metrology and open-set similarity gating to identify fish species, measure curved total length in millimeters, estimate allometric biomass, and flag out-of-distribution specimens for human review before persisting records to DuckDB.

---

## Architecture

```text
┌───────────────────┐     HTTP      ┌──────────────────────────────────────────────┐
│  Streamlit UI     │──────────────▶│              FastAPI Backend                 │
│  app/dashboard.py │               │         src/catchvision/api/main.py          │
│  (:8501)          │◀──────────────│  gpu_lock │ db_lock │ PendingInspection HIL  │
└───────────────────┘               └──────┬───────────────────────────────┬───────┘
                                           │                               │
                 ┌─────────────────────────┼───────────────────┐           │
                 ▼                         ▼                   ▼           ▼
       [1. Vision Stage]         [2. Metrology Stage]  [3. Open-Set ID]  [Storage]
       GroundingDINO-tiny        ArUco DICT_4X4_50     BioCLIP           DuckDB
       + SAM 2.1 (sam2.1_b)      Homography + Geodesic Cosine OOD Gate   survey_frames
       -> Neutral Gray Crops     Midline TL & Depth    + LWR Biomass     catch_records
                                                                         benchmark_runs
                 │
                 ▼  (Eval & Stress Suites)
       [ExperimentTracker] ──▶ Local JSON Manifests + MLflow Tracking (:5000)
```

### Core Pipeline Stages

1. **Detection & Segmentation (`src/catchvision/vision/segmentor.py`):**
   * Runs `IDEA-Research/grounding-dino-tiny` (`box_threshold=0.28`) with IoU and containment suppression to eliminate duplicate or fin-level sub-boxes.
   * Extracts instance masks with Ultralytics SAM 2.1 (`sam2.1_b.pt`) and generates neutral-gray (`128`) background-masked crops to prevent reef/dock background leakage into classification embeddings.
2. **Perspective Calibration & Metric Sizing (`src/catchvision/geometry/calibration.py`):**
   * Detects corner markers on a printable A4 landscape ArUco mat (`DICT_4X4_50`, `297 × 210 mm`, `35 mm` markers; generator in `aruco_board.py`).
   * Supports automatic fallback to marker-corner correspondences when 1–2 markers are occluded (`min_markers=2`).
   * Computes curved **Total Length** via distance-transform-biased geodesic midline tracing (`skimage.graph.route_through_array`), plus straight Feret length, body depth, and projected area ($\text{cm}^2$).
3. **Open-Set BioCLIP Identification & Allometry (`src/catchvision/vision/classifier.py`, `biology/metrics.py`):**
   * Encodes taxonomic and morphological prompts for 15 Red Sea reef species using `hf-hub:imageomics/bioclip`.
   * Applies a dual-gate Human-in-the-Loop (HIL) policy (`min_conf_for_auto=0.70`, `min_margin_for_auto=0.25`, `min_cosine_sim=0.22`) to prevent softmax overconfidence on non-catalog/OOD species.
   * Computes specimen weight via species-specific Length-Weight Relationships ($W = a \cdot L^b$) and checks Minimum Legal Size (MLS) compliance.
4. **Field Stress Benchmarking & MLOps (`benchmark/stress.py`, `storage/catch_db.py`, `mlops/tracking.py`):**
   * Evaluates sizing and ID stability under synthetic dockside degradations: `sun_glare`, `water_droplets`, `low_light`, `motion_blur`, `marker_occlusion_1`, and `marker_occlusion_2`.
   * Logs catch records and benchmark aggregates in DuckDB, and tracks evaluation metrics/CSV artifacts in local JSON manifests and MLflow.

---

## Experimental Results

### 1. Open-Set Species ID & OOD Gating (`scripts/eval_redsea.py`, CUDA)

Evaluated on unconstrained underwater iNaturalist research-grade photos across **10 target Red Sea species** (30 frames) and **2 OOD control species** (*Diodon hystrix*, *Arothron stellatus*):

| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Detection Rate (In-Dist)** | **90.0%** (27/30) | GroundingDINO (`box_threshold=0.28`) + SAM 2.1 |
| **Zero-Shot Top-1 / Top-3** | **48.1% / 70.4%** | Unconstrained angles & underwater lighting |
| **In-Dist Raw Cosine Sim** | **0.255** (median) | BioCLIP unscaled cosine similarity |
| **OOD Max Raw Cosine Sim** | **0.2031** | *Diodon hystrix* (`0.2010`), *Arothron stellatus* (`0.2031`) |
| **OOD False Softmax Conf** | `0.6197 – 0.8136` | High softmax despite OOD; gated cleanly by `min_cosine_sim=0.22` |
| **1-Shot Prototype Ablation** | `35.3%` (vs `47.1%` ZS) | Single unconstrained support crop introduces pose/lighting bias |

> **Dockside OOD Case Study:** On a camouflaged *Arothron stellatus* test frame (detected at `box_threshold=0.21`), closed-set BioCLIP softmax predicted *Epinephelus polyphekadion* with `0.99` confidence and `0.99` margin, while raw cosine similarity was only `0.205` — triggering the `Review` flag at the `0.22` OOD gate.

### 2. Field Stress Benchmark (`synthetic_tilted.jpg`, 150.0 mm GT TL)

Evaluated across synthetic field perturbations at severities `0.4` and `0.8`:

| Metric | Result | Target |
| :--- | :---: | :---: |
| **Calibration Success Rate** | **100%** | ≥ 90% (including 2 occluded markers) |
| **Mean Length Error (MAPE)** | **0.41%** (`~0.6 mm` MAE) | < 2.0% |
| **Species Consistency** | **100%** | Across tested low-light & occlusion runs |

---

## Repository Structure

```text
RedSea-CatchVision/
├── app/
│   ├── dashboard.py              # Thin Streamlit UI (Dock Inspection, Ledger, Benchmark)
│   └── Dockerfile
├── scripts/
│   └── eval_redsea.py            # iNaturalist zero-shot, 1-shot & OOD benchmark runner
├── notebooks/
│   └── field_experiment_eval.ipynb
├── src/catchvision/
│   ├── api/
│   │   ├── main.py               # FastAPI inference & DuckDB ledger service
│   │   ├── schemas.py            # Pydantic API contracts
│   │   └── Dockerfile
│   ├── benchmark/
│   │   └── stress.py             # Synthetic dockside degradation benchmark suite
│   ├── biology/
│   │   └── metrics.py            # Allometric W = a * L^b and Fulton's condition factor K
│   ├── geometry/
│   │   ├── aruco_board.py        # Printable A4 ArUco calibration mat generator
│   │   └── calibration.py        # Homography rectification & geodesic midline TL
│   ├── mlops/
│   │   └── tracking.py           # JSON manifest & MLflow experiment tracker
│   ├── storage/
│   │   └── catch_db.py           # Analytical DuckDB store & SQL aggregations
│   ├── vision/
│   │   ├── classifier.py         # BioCLIP zero-shot/few-shot ID + OOD gating + LWR
│   │   └── segmentor.py          # GroundingDINO + SAM 2.1 segmentation
│   └── config.py                 # Unified Pydantic Settings (CATCHVISION_* env vars)
├── tests/                        # Unit & API integration tests (pytest)
├── docker-compose.yml            # Multi-service stack: mlflow, api, ui
└── pyproject.toml
```

---

## Quickstart

### Option A: Docker Compose (Recommended)

```bash
git clone [https://github.com/](https://github.com/)DanilKras/RedSea_CatchVision.git
cd RedSea_CatchVision

docker compose up --build
```

Services will be available at:
* **Streamlit Dashboard:** http://localhost:8501
* **FastAPI Swagger Docs:** http://localhost:8000/docs
* **MLflow Tracking UI:** http://localhost:5000

### Option B: Local Development (`uv`)

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync

# 1. Start MLflow server (optional, or use local SQLite URI)
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination ./mlruns/artifacts --serve-artifacts

# 2. Start Inference API (single worker for shared GPU/DuckDB locks)
MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run uvicorn catchvision.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Start Streamlit Dashboard
uv run streamlit run app/dashboard.py
```

### Run Evaluation & Tests

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/eval_redsea.py

# Generate printable A4 ArUco calibration mat
uv run python -m catchvision.geometry.aruco_board

# Run test suite & linter
uv run pytest
uv run ruff check .
```

---

## Configuration

All runtime thresholds and paths are centralized in `src/catchvision/config.py` (`Settings`) and can be overridden via `CATCHVISION_*` environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CATCHVISION_BOX_THRESHOLD` | `0.28` | GroundingDINO detection confidence threshold |
| `CATCHVISION_MIN_COSINE_SIM` | `0.22` | BioCLIP raw cosine OOD gating threshold |
| `CATCHVISION_MIN_CONF_FOR_AUTO` | `0.70` | Minimum softmax confidence for auto-commit |
| `CATCHVISION_MIN_MARGIN_FOR_AUTO` | `0.25` | Minimum Top-1 vs Top-2 probability margin |
| `CATCHVISION_ARUCO_MIN_MARKERS` | `2` | Minimum visible A4 markers for calibration |
| `CATCHVISION_SKIP_MODEL_LOAD` | `false` | Skip loading CV weights (used in CI/unit tests) |

---

## Engineering Trade-offs & Roadmap

* **Stage 1 Segmentation (`GroundingDINO + SAM 2.1` vs `SAM 3`): `GroundingDINO-tiny + SAM 2.1` provides zero-config reproducibility for reviewers without requiring `HF_TOKEN` approval; swapping Stage 1 for SAM 3 is a drop-in change inside `FishSegmentor`.
* **Visual Prototypes:** Unconstrained 1-shot underwater prototypes degraded Top-1 accuracy (`47.1% → 35.3%`) due to background and pose variance. Multi-shot prototype fitting (`classifier.fit_visual_prototypes`) should be calibrated on standardized dockside mat photos (`text_weight >= 0.70`).