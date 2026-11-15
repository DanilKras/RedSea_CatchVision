# RedSea-CatchVision

**Open-Set Reef Fishery Monitoring: Zero-Shot Segmentation, ArUco Metric Sizing & Allometric Biomass Estimation**

`RedSea-CatchVision` is an end-to-end computer vision and telemetry pipeline built for artisanal reef fishery landing surveys in the Red Sea. It combines foundation vision models (`GroundingDINO` + `SAM 2.1` + `BioCLIP`) with perspective ArUco metrology and open-set OOD gating to identify reef species, measure curved total length (`mm`), estimate biomass ($W = a \cdot L^b$), and flag uncertain catches for human review before logging to `DuckDB`.

---

## Key Capabilities

1. **Zero-Shot Detection & Segmentation (`GroundingDINO-tiny` + `SAM 2.1`):**
   - Detects fish instances and extracts neutral-gray background-masked crops to prevent dockside background leakage into classification embeddings.
2. **Perspective Calibration & Geodesic Sizing (`ArUco`):**
   - Rectifies camera perspective using a printable A4 ArUco mat (`DICT_4X4_50`, works even with 1–2 occluded markers) and computes curved **Total Length** via geodesic midline tracing (`0.41%` MAPE / `~0.6 mm` error).
3. **Open-Set BioCLIP Identification & Biomass Allometry:**
   - Classifies 15 Red Sea reef species with `imageomics/bioclip` and enforces a raw cosine similarity gate (`min_cosine_sim=0.22`) to catch out-of-distribution (OOD) species before computing Length-Weight biomass and Minimum Legal Size (MLS) compliance.
4. **Full-Stack Telemetry & MLOps (`FastAPI`, `Streamlit`, `DuckDB`, `MLflow`):**
   - Provides a dockside inspection UI, Human-in-the-Loop review queue, synthetic field stress benchmarking (glare, low light, blur, marker occlusion), and experiment tracking.

---

## Experimental Highlights

### 1. Open-Set Species ID & OOD Gating (Red Sea iNaturalist Benchmark)
* **Detection Rate:** **90.0%** (27/30 frames across 10 target Red Sea species).
* **Zero-Shot Accuracy:** **48.1% Top-1 / 70.4% Top-3** on unconstrained underwater imagery.
* **OOD Rejection:** Non-catalog species (*Diodon hystrix*, *Arothron stellatus*) scored `≤ 0.2031` raw cosine similarity and were cleanly routed to human review (`min_cosine_sim=0.22`), preventing false high-confidence softmax predictions.

### 2. Dockside Stress Benchmark (`150.0 mm` Ground Truth)
* **Calibration Success Rate:** **100%** (including runs with 2 occluded ArUco markers).
* **Mean Length Error (MAPE):** **0.41%** (`~0.6 mm` MAE) across low-light, blur, and occlusion perturbations.

---

## Project Structure

```text
RedSea_CatchVision/
├── app/
│   └── dashboard.py            # Streamlit UI: Dock Inspection, Catch Ledger & Stress Benchmark
├── scripts/
│   └── eval_redsea.py          # Zero-shot, 1-shot & OOD evaluation script
├── src/catchvision/
│   ├── api/                    # FastAPI inference endpoints & DuckDB ledger service
│   ├── benchmark/              # Synthetic dockside degradation suite
│   ├── biology/                # Allometric W = a * L^b & Fulton's condition factor K
│   ├── geometry/               # A4 ArUco mat generator, homography & geodesic midline TL
│   ├── mlops/                  # JSON manifest & MLflow experiment tracker
│   ├── storage/                # DuckDB analytical store & SQL aggregations
│   ├── vision/                 # GroundingDINO + SAM 2.1 segmentor & BioCLIP OOD classifier
│   └── config.py               # Centralized Pydantic settings (CATCHVISION_* env vars)
├── tests/                      # Pytest unit & FastAPI integration suite
├── docker-compose.yml
└── pyproject.toml
```

---

## Quickstart

### Option 1: Docker Compose (Recommended)

```bash
git clone [https://github.com/DanilKras/RedSea_CatchVision.git](https://github.com/DanilKras/RedSea_CatchVision.git)
cd RedSea_CatchVision
docker compose up --build
```

* **Streamlit Dashboard:** `http://localhost:8501`
* **FastAPI Swagger Docs:** `http://localhost:8000/docs`
* **MLflow Tracking UI:** `http://localhost:5000`

### Option 2: Local Development with `uv`

```bash
# 1. Install dependencies
uv sync

# 2. Run unit and API tests
uv run pytest

# 3. Start API and dashboard
uv run uvicorn catchvision.api.main:app --port 8000 --workers 1 &
uv run streamlit run app/dashboard.py
```
