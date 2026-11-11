from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from catchvision.biology.metrics import estimate_weight_g
from catchvision.config import Settings

DEFAULT_BIOCLIP_MODEL = "hf-hub:imageomics/bioclip"


class SpeciesSpec(BaseModel):
    """Reference card for a Red Sea fish species used in zero-shot ID and biomass estimation."""

    scientific_name: str
    common_name: str
    local_name_ar: str = ""
    family: str = ""
    trophic_guild: str = "reef_associated"
    iucn_status: str = "LC"
    # Length-weight relationship parameters W(g) = a * L(cm)^b
    lwr_a: float | None = None
    lwr_b: float | None = None
    min_legal_length_cm: float | None = None


class SpeciesPrediction(BaseModel):
    """Single candidate prediction inside Top-K results."""

    scientific_name: str
    common_name: str
    local_name_ar: str
    family: str
    probability: float
    cosine_sim: float = 0.0


class ClassificationResult(BaseModel):
    """Final species prediction and biological metrics for a cropped fish specimen."""

    fish_id: int
    top_species: str
    common_name: str
    local_name_ar: str
    family: str
    confidence: float
    margin_top1_top2: float
    raw_cosine_sim: float = 0.0
    low_cosine_similarity: bool = False
    used_visual_prototypes: bool = False
    estimated_weight_g: float | None = None
    below_min_legal_size: bool | None = None
    top_k: list[SpeciesPrediction] = Field(default_factory=list)


DEFAULT_RED_SEA_SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec(
        scientific_name="Plectropomus pessuliferus",
        common_name="Roving coral grouper",
        local_name_ar="Najil",
        family="Serranidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0123,
        lwr_b=3.04,
        min_legal_length_cm=35.0,
    ),
    SpeciesSpec(
        scientific_name="Variola louti",
        common_name="Yellow-edged lyretail",
        local_name_ar="Shaer",
        family="Serranidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0138,
        lwr_b=3.02,
        min_legal_length_cm=30.0,
    ),
    SpeciesSpec(
        scientific_name="Cephalopholis miniata",
        common_name="Coral hind",
        local_name_ar="Hamour ahmar",
        family="Serranidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0155,
        lwr_b=3.01,
        min_legal_length_cm=25.0,
    ),
    SpeciesSpec(
        scientific_name="Epinephelus polyphekadion",
        common_name="Camouflage grouper",
        local_name_ar="Kushar",
        family="Serranidae",
        trophic_guild="piscivore",
        iucn_status="VU",
        lwr_a=0.0141,
        lwr_b=3.02,
        min_legal_length_cm=35.0,
    ),
    SpeciesSpec(
        scientific_name="Lutjanus bohar",
        common_name="Two-spot red snapper",
        local_name_ar="Bohar",
        family="Lutjanidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0162,
        lwr_b=2.98,
        min_legal_length_cm=30.0,
    ),
    SpeciesSpec(
        scientific_name="Lethrinus nebulosus",
        common_name="Spangled emperor",
        local_name_ar="Shaoor",
        family="Lethrinidae",
        trophic_guild="invertivore",
        iucn_status="LC",
        lwr_a=0.0178,
        lwr_b=2.97,
        min_legal_length_cm=28.0,
    ),
    SpeciesSpec(
        scientific_name="Scarus ferrugineus",
        common_name="Rusty parrotfish",
        local_name_ar="Hareed",
        family="Scaridae",
        trophic_guild="herbivore",
        iucn_status="LC",
        lwr_a=0.0191,
        lwr_b=2.99,
        min_legal_length_cm=22.0,
    ),
    SpeciesSpec(
        scientific_name="Siganus rivulatus",
        common_name="Marbled spinefoot rabbitfish",
        local_name_ar="Sijan",
        family="Siganidae",
        trophic_guild="herbivore",
        iucn_status="LC",
        lwr_a=0.0115,
        lwr_b=3.05,
        min_legal_length_cm=18.0,
    ),
    SpeciesSpec(
        scientific_name="Acanthurus sohal",
        common_name="Sohal surgeonfish",
        local_name_ar="Sohal",
        family="Acanthuridae",
        trophic_guild="herbivore",
        iucn_status="LC",
        lwr_a=0.0210,
        lwr_b=2.95,
        min_legal_length_cm=20.0,
    ),
    SpeciesSpec(
        scientific_name="Cheilinus undulatus",
        common_name="Humphead wrasse",
        local_name_ar="Tarabani",
        family="Labridae",
        trophic_guild="invertivore",
        iucn_status="EN",
        lwr_a=0.0166,
        lwr_b=3.00,
        min_legal_length_cm=50.0,
    ),
    SpeciesSpec(
        scientific_name="Pterois miles",
        common_name="Devil firefish",
        local_name_ar="Dajajet al-bahr",
        family="Scorpaenidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0108,
        lwr_b=3.06,
    ),
    SpeciesSpec(
        scientific_name="Naso unicornis",
        common_name="Bluespine unicornfish",
        local_name_ar="Abu qarn",
        family="Acanthuridae",
        trophic_guild="herbivore",
        iucn_status="LC",
        lwr_a=0.0182,
        lwr_b=2.98,
        min_legal_length_cm=25.0,
    ),
    SpeciesSpec(
        scientific_name="Caranx sexfasciatus",
        common_name="Bigeye trevally",
        local_name_ar="Bayad",
        family="Carangidae",
        trophic_guild="piscivore",
        iucn_status="LC",
        lwr_a=0.0195,
        lwr_b=2.94,
        min_legal_length_cm=30.0,
    ),
    SpeciesSpec(
        scientific_name="Pomacanthus maculosus",
        common_name="Yellowbar angelfish",
        local_name_ar="Anfaz",
        family="Pomacanthidae",
        trophic_guild="invertivore",
        iucn_status="LC",
        lwr_a=0.0251,
        lwr_b=2.92,
    ),
    SpeciesSpec(
        scientific_name="Amphiprion bicinctus",
        common_name="Two-band anemonefish",
        local_name_ar="Samak al-bahr",
        family="Pomacentridae",
        trophic_guild="omnivore",
        iucn_status="LC",
    ),
)


class FishClassifier:
    def __init__(
        self,
        species_catalog: Sequence[SpeciesSpec] | None = None,
        model_id: str | None = None,
        settings: Settings | None = None,
        device: str | None = None,
        text_weight: float | None = None,
        min_cosine_sim: float | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or getattr(
            self.settings,
            "bioclip_model_id",
            DEFAULT_BIOCLIP_MODEL,
        )
        tw = self.settings.bioclip_text_weight if text_weight is None else text_weight
        self.text_weight = float(np.clip(tw, 0.0, 1.0))
        self.min_cosine_sim = (
            self.settings.min_cosine_sim if min_cosine_sim is None else min_cosine_sim
        )

        self.species: list[SpeciesSpec] = list(species_catalog or DEFAULT_RED_SEA_SPECIES)
        if not self.species:
            raise ValueError("species_catalog must contain at least one SpeciesSpec.")
        self._species_index = {spec.scientific_name: i for i, spec in enumerate(self.species)}

        logger.info(f"Loading BioCLIP ({self.model_id}) on device={self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_id)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        self._text_embeddings = self._encode_species_catalog(self.species)
        self._visual_prototypes: dict[str, torch.Tensor] = {}

    @staticmethod
    def _build_prompts(spec: SpeciesSpec) -> list[str]:
        """Build taxonomic and visual prompt templates tailored for BioCLIP."""
        sci = spec.scientific_name
        common = spec.common_name
        fam = spec.family
        prompts = [
            f"a photo of a {common}, scientific name {sci}",
            f"a close-up biological specimen photo of {sci} reef fish",
            f"a lateral view of {common} ({sci}) on a plain background",
            f"a Red Sea coral reef fish {sci}",
        ]
        if fam:
            prompts.append(f"Animalia Chordata Actinopterygii {fam} {sci} {common}")
        return prompts

    @torch.inference_mode()
    def _encode_species_catalog(self, catalog: Sequence[SpeciesSpec]) -> torch.Tensor:
        """Encode the species catalog into an L2-normalized embedding matrix [N_species, D]."""
        class_vectors: list[torch.Tensor] = []
        for spec in catalog:
            prompts = self._build_prompts(spec)
            tokens = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
            mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
            class_vectors.append(mean_feat.squeeze(0))
        return torch.stack(class_vectors, dim=0)

    def set_species_catalog(self, catalog: Sequence[SpeciesSpec]) -> None:
        """Update candidate species without reloading BioCLIP weights."""
        if not catalog:
            raise ValueError("species_catalog cannot be empty.")
        self.species = list(catalog)
        self._species_index = {spec.scientific_name: i for i, spec in enumerate(self.species)}
        self._text_embeddings = self._encode_species_catalog(self.species)

    @staticmethod
    def _to_pil(image: Path | str | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Path | str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[Path | str | Image.Image | np.ndarray],
    ) -> torch.Tensor:
        """Extract L2-normalized BioCLIP image embeddings [N, D] for a batch of crops."""
        tensors = [self.preprocess(self._to_pil(img)) for img in images]
        batch = torch.stack(tensors, dim=0).to(self.device)
        feats = self.model.encode_image(batch)
        return F.normalize(feats, dim=-1)

    @torch.inference_mode()
    def fit_visual_prototypes(
        self,
        samples_by_species: dict[str, Sequence[Path | str | Image.Image | np.ndarray]],
    ) -> int:
        """
        Compute few-shot visual prototypes from reference images for hybrid classification.
        Dictionary keys must match `scientific_name` entries in the active catalog.
        """
        loaded = 0
        for sci_name, images in samples_by_species.items():
            if sci_name not in self._species_index or not images:
                continue
            feats = self.encode_images(images)
            proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            self._visual_prototypes[sci_name] = proto.detach().cpu()
            loaded += 1
        logger.info(f"Fitted visual BioCLIP prototypes for {loaded} species.")
        return loaded

    def save_prototypes(self, path: Path) -> None:
        """Save fitted visual prototype vectors to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "prototypes": {k: v.tolist() for k, v in self._visual_prototypes.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_prototypes(self, path: Path) -> int:
        """Load saved visual prototype vectors from JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0
        for sci_name, vec in payload.get("prototypes", {}).items():
            if sci_name in self._species_index:
                tensor = torch.tensor(vec, dtype=torch.float32)
                self._visual_prototypes[sci_name] = F.normalize(tensor, dim=0)
                loaded += 1
        logger.info(f"Loaded {loaded} visual prototypes from {path}")
        return loaded

    def _effective_class_embeddings(self) -> tuple[torch.Tensor, bool]:
        """Blend text prompt embeddings with visual prototypes when available."""
        if not self._visual_prototypes:
            return self._text_embeddings, False

        combined = self._text_embeddings.clone()
        for sci_name, proto_cpu in self._visual_prototypes.items():
            idx = self._species_index.get(sci_name)
            if idx is None:
                continue
            proto = proto_cpu.to(device=combined.device, dtype=combined.dtype)
            blended = self.text_weight * combined[idx] + (1.0 - self.text_weight) * proto
            combined[idx] = F.normalize(blended, dim=0)
        return combined, True

    @staticmethod
    def _estimate_biomass(
        spec: SpeciesSpec,
        length_cm: float | None,
    ) -> tuple[float | None, bool | None]:
        if length_cm is None or length_cm <= 0:
            return None, None
        weight_g = None
        if spec.lwr_a is not None and spec.lwr_b is not None:
            weight_g = round(estimate_weight_g(length_cm, spec.lwr_a, spec.lwr_b), 2)
        below_legal = None
        if spec.min_legal_length_cm is not None:
            below_legal = bool(length_cm < spec.min_legal_length_cm)
        return weight_g, below_legal

    @torch.inference_mode()
    def classify_batch(
        self,
        images: Sequence[Path | str | Image.Image | np.ndarray],
        fish_ids: Sequence[int] | None = None,
        lengths_cm: Sequence[float | None] | None = None,
        top_k: int = 5,
    ) -> list[ClassificationResult]:
        if not images:
            return []

        ids = list(fish_ids) if fish_ids is not None else list(range(len(images)))
        lens = list(lengths_cm) if lengths_cm is not None else [None] * len(images)
        if not (len(ids) == len(images) == len(lens)):
            raise ValueError("images, fish_ids, and lengths_cm must have the same length.")

        img_feats = self.encode_images(images)
        class_feats, used_proto = self._effective_class_embeddings()

        cos_sims = img_feats @ class_feats.T
        logit_scale = float(self.model.logit_scale.exp().clamp(max=100.0).item())
        logits = cos_sims * logit_scale
        probs = logits.softmax(dim=-1).cpu().numpy()
        cos_np = cos_sims.cpu().numpy()

        k = min(top_k, len(self.species))
        results: list[ClassificationResult] = []
        for fish_id, row_probs, row_cos, length_cm in zip(ids, probs, cos_np, lens, strict=True):
            order = np.argsort(-row_probs)[:k]
            top_idx = int(order[0])
            second_prob = float(row_probs[order[1]]) if len(order) > 1 else 0.0
            top_cos = float(row_cos[top_idx])
            top_spec = self.species[top_idx]
            weight_g, below_legal = self._estimate_biomass(top_spec, length_cm)

            top_preds = [
                SpeciesPrediction(
                    scientific_name=self.species[int(idx)].scientific_name,
                    common_name=self.species[int(idx)].common_name,
                    local_name_ar=self.species[int(idx)].local_name_ar,
                    family=self.species[int(idx)].family,
                    probability=round(float(row_probs[int(idx)]), 4),
                    cosine_sim=round(float(row_cos[int(idx)]), 4),
                )
                for idx in order
            ]

            results.append(
                ClassificationResult(
                    fish_id=fish_id,
                    top_species=top_spec.scientific_name,
                    common_name=top_spec.common_name,
                    local_name_ar=top_spec.local_name_ar,
                    family=top_spec.family,
                    confidence=round(float(row_probs[top_idx]), 4),
                    margin_top1_top2=round(float(row_probs[top_idx] - second_prob), 4),
                    raw_cosine_sim=round(top_cos, 4),
                    low_cosine_similarity=bool(top_cos < self.min_cosine_sim),
                    used_visual_prototypes=used_proto,
                    estimated_weight_g=weight_g,
                    below_min_legal_size=below_legal,
                    top_k=top_preds,
                )
            )
        return results

    def classify(
        self,
        image: Path | str | Image.Image | np.ndarray,
        fish_id: int = 0,
        length_cm: float | None = None,
        top_k: int = 5,
    ) -> ClassificationResult:
        """Classify a single fish crop (raw or background-masked)."""
        return self.classify_batch(
            images=[image],
            fish_ids=[fish_id],
            lengths_cm=[length_cm],
            top_k=top_k,
        )[0]
