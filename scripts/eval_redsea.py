import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from catchvision.config import Settings
from loguru import logger

from catchvision.mlops.tracking import ExperimentTracker
from catchvision.vision.classifier import FishClassifier
from catchvision.vision.segmentor import FishSegmentor

IN_DIST_SPECIES = (
    "Plectropomus pessuliferus",
    "Variola louti",
    "Cephalopholis miniata",
    "Lutjanus bohar",
    "Lethrinus nebulosus",
    "Scarus ferrugineus",
    "Siganus rivulatus",
    "Acanthurus sohal",
    "Cheilinus undulatus",
    "Pterois miles",
)
OOD_SPECIES = ("Diodon hystrix", "Arothron stellatus")

SETTINGS = Settings()
EVAL_DIR = SETTINGS.data_dir / "eval_redsea"
PROC_DIR = SETTINGS.eval_redsea_dir


def fetch_inat_photos(species, out_dir, limit=3):
    sp_dir = out_dir / species.replace(" ", "_")
    sp_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(sp_dir.glob("*.jpg"))
    if len(existing) >= limit:
        return existing[:limit]

    params = urllib.parse.urlencode(
        {
            "taxon_name": species,
            "quality_grade": "research",
            "photos": "true",
            "per_page": limit * 3,
            "order_by": "votes",
        }
    )
    url = f"https://api.inaturalist.org/v1/observations?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "CatchVision/0.1"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            results = json.loads(resp.read().decode()).get("results", [])
    except Exception as exc:
        logger.warning(f"API error for {species}: {exc}")
        return existing

    saved = list(existing)
    for obs in results:
        if len(saved) >= limit:
            break
        photos = obs.get("photos") or []
        if not photos:
            continue

        img_url = photos[0].get("url", "").replace("square.", "medium.")
        dst = sp_dir / f"inat_{obs['id']}.jpg"
        if not img_url or dst.exists():
            continue

        try:
            img_req = urllib.request.Request(img_url, headers={"User-Agent": "CatchVision/0.1"})
            with urllib.request.urlopen(img_req, timeout=20) as resp:
                dst.write_bytes(resp.read())
            saved.append(dst)
            time.sleep(0.25)
        except Exception as exc:
            logger.warning(f"Download failed ({img_url}): {exc}")

    return saved[:limit]


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    segmentor = FishSegmentor(settings=SETTINGS)
    classifier = FishClassifier(settings=SETTINGS)

    targets = [(sp, False) for sp in IN_DIST_SPECIES] + [(sp, True) for sp in OOD_SPECIES]
    records = []

    for species, is_ood in targets:
        photos = fetch_inat_photos(species, EVAL_DIR, limit=3)
        logger.info(f"{species} (OOD={is_ood}): {len(photos)} photos")
        run_out = PROC_DIR / species.replace(" ", "_")

        for img_path in photos:
            items, _ = segmentor.segment_image(img_path, run_out, save_overlay=False)
            if not items:
                records.append(
                    {
                        "gt_species": species,
                        "is_ood": is_ood,
                        "image": img_path.name,
                        "detected": False,
                        "pred_species": None,
                        "top1_ok": False,
                        "top3_ok": False,
                        "cls_conf": np.nan,
                        "raw_cosine_sim": np.nan,
                        "crop_path": None,
                    }
                )
                continue

            best = max(items, key=lambda it: it.mask_area_px * it.confidence)
            res = classifier.classify(best.masked_crop_path, fish_id=best.fish_id, top_k=3)
            top3 = [p.scientific_name for p in res.top_k]

            records.append(
                {
                    "gt_species": species,
                    "is_ood": is_ood,
                    "image": img_path.name,
                    "detected": True,
                    "pred_species": res.top_species,
                    "top1_ok": (res.top_species == species) if not is_ood else False,
                    "top3_ok": (species in top3) if not is_ood else False,
                    "cls_conf": res.confidence,
                    "raw_cosine_sim": res.raw_cosine_sim,
                    "crop_path": str(best.masked_crop_path),
                }
            )

    df = pd.DataFrame(records)
    csv_path = PROC_DIR / "eval_zero_shot.csv"
    df.drop(columns=["crop_path"]).to_csv(csv_path, index=False)

    in_df = df[~df["is_ood"]]
    in_det = in_df[in_df["detected"]]
    ood_det = df[df["is_ood"] & df["detected"]]
    det_rate = len(in_det) / max(len(in_df), 1)

    print(f"\nDetected: {len(in_det)}/{len(in_df)} ({det_rate:.1%})")
    if not in_det.empty:
        print(f"Top-1: {in_det['top1_ok'].mean():.1%} | Top-3: {in_det['top3_ok'].mean():.1%}")
        print(
            f"In-dist cosine min={in_det['raw_cosine_sim'].min():.3f}, median={in_det['raw_cosine_sim'].median():.3f}"
        )
    if not ood_det.empty:
        print(f"OOD cosine max={ood_det['raw_cosine_sim'].max():.3f}")
        print(
            ood_det[["gt_species", "pred_species", "cls_conf", "raw_cosine_sim"]].to_string(
                index=False
            )
        )

    metrics = {
        "detection_rate": float(det_rate),
        "top1_all": float(in_det["top1_ok"].mean()) if not in_det.empty else 0.0,
        "top3_all": float(in_det["top3_ok"].mean()) if not in_det.empty else 0.0,
    }
    if not ood_det.empty:
        metrics["ood_max_cos"] = float(ood_det["raw_cosine_sim"].max())

    support, queries = {}, []
    for sp, grp in in_det.groupby("gt_species"):
        if len(grp) >= 2:
            paths = [Path(p) for p in grp["crop_path"]]
            support[sp] = [paths[0]]
            for p, zs_ok in zip(paths[1:], grp["top1_ok"].iloc[1:]):
                queries.append((sp, p, zs_ok))

    if support and queries:
        classifier.fit_visual_prototypes(support)
        one_shot_hits = [classifier.classify(p, top_k=3).top_species == sp for sp, p, _ in queries]
        metrics["zero_shot_query_top1"] = float(np.mean([zs_ok for _, _, zs_ok in queries]))
        metrics["one_shot_query_top1"] = float(np.mean(one_shot_hits))
        print(
            f"\n1-Shot ({len(queries)} queries): "
            f"Zero-Shot {metrics['zero_shot_query_top1']:.1%} -> 1-Shot {metrics['one_shot_query_top1']:.1%}"
        )

    ExperimentTracker().log_run(
        run_name="inat_open_set_eval",
        params={
            "segmentor_backend": segmentor.settings.segmentor_backend,
            "box_threshold": segmentor.settings.box_threshold,
            "min_cosine_sim": classifier.min_cosine_sim,
            "bioclip_model": classifier.model_id,
        },
        metrics=metrics,
        artifacts=[csv_path],
    )


if __name__ == "__main__":
    main()
