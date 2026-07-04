from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEFAULT_TEXT_MODEL_ID = "google/siglip2-base-patch32-256"
DEFAULT_ONNX_MODEL = "siglip2-base-patch32-256_vision_encoder.sim.onnx"
DEFAULT_PROMPT_FILE = "jewelry_prompts.json"
DEFAULT_CACHE_FILE = "jewelry_text_embeddings_cache.npz"
DEFAULT_GALLERY_FILE = "jewelry_correction_gallery.npz"
GALLERY_THRESHOLD = 0.90
WHITE_THRESHOLD = 245
FOREGROUND_PADDING_RATIO = 0.12
SIMILARITY_SCALE = 100.0
GOLD_TYPE_MIN_SIMILARITY = 0.08
GOLD_TYPE_REJECTION_MARGIN = 0.03
PROMPT_EXPANSION_VERSION = "white-tabletop-v1"
PRODUCT_CONTEXT_VARIANTS = (
    "",
    "laid flat on a plain white background on a table",
    "isolated studio product photo, not worn on the body",
)


@dataclass
class ScoreEntry:
    label: str
    similarity: float
    confidence: float


@dataclass(frozen=True)
class ClassPromptDefinition:
    label: str
    group: str
    prompts: list[str]


@dataclass
class PredictionResult:
    label: str
    confidence: float
    scores: list[ScoreEntry]
    image_path: str | None
    original_image: Image.Image
    cropped_image: Image.Image
    gallery_match: bool = False
    gallery_similarity: float = 0.0
    is_gold_jewelry: bool = True
    gold_verification_reason: str = ""


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return vector / norm


class CorrectionGallery:
    def __init__(self, gallery_path: Path) -> None:
        self.gallery_path = gallery_path
        self.embeddings: np.ndarray | None = None
        self.labels: list[str] = []
        self.load()

    def load(self) -> None:
        if self.gallery_path.exists():
            try:
                with np.load(self.gallery_path, allow_pickle=True) as data:
                    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                    labels = [str(label) for label in data["labels"].tolist()]

                if embeddings.size == 0:
                    self.embeddings = None
                    self.labels = []
                    return
                if embeddings.ndim == 1:
                    embeddings = embeddings[np.newaxis, :]
                if embeddings.ndim != 2 or embeddings.shape[0] != len(labels):
                    raise ValueError("Gallery embeddings and labels have incompatible shapes.")

                self.embeddings = _normalize_vector(embeddings)
                self.labels = labels
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to load correction gallery: {exc}")
                self.embeddings = None
                self.labels = []

    def save(self) -> None:
        if self.embeddings is not None:
            try:
                self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    self.gallery_path,
                    embeddings=self.embeddings,
                    labels=np.array(self.labels),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to save correction gallery: {exc}")

    def add(self, embedding: np.ndarray, label: str) -> None:
        normalized_label = str(label).strip()
        if not normalized_label:
            raise ValueError("Gallery label cannot be empty.")

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError("Gallery embedding must contain exactly one vector.")
        vector = _normalize_vector(vector)[np.newaxis, :]

        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = vector
        else:
            if self.embeddings.shape[1] != vector.shape[1]:
                raise ValueError("Gallery embedding dimension does not match existing entries.")
            self.embeddings = np.vstack([self.embeddings, vector])

        self.labels.append(normalized_label)
        self.save()

    def search(self, embedding: np.ndarray, threshold: float = GALLERY_THRESHOLD) -> tuple[str | None, float]:
        if self.embeddings is None or len(self.labels) == 0:
            return None, 0.0

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match the gallery.")
        vector = _normalize_vector(vector)

        similarities = self.embeddings @ vector
        idx = int(np.argmax(similarities))
        score = float(similarities[idx])

        if score >= threshold:
            return self.labels[idx], score
        return None, score


class JewelryZeroShotClassifier:
    def __init__(
        self,
        onnx_model_path: str | Path = DEFAULT_ONNX_MODEL,
        prompt_path: str | Path = DEFAULT_PROMPT_FILE,
        text_model_id: str = DEFAULT_TEXT_MODEL_ID,
        embedding_cache_path: str | Path = DEFAULT_CACHE_FILE,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.onnx_model_path = self._resolve_path(onnx_model_path)
        self.prompt_path = self._resolve_path(prompt_path)
        self.embedding_cache_path = self._resolve_path(embedding_cache_path)
        self.gallery_path = self.base_dir / DEFAULT_GALLERY_FILE
        self.text_model_id = text_model_id
        (
            self.group_prompt_config,
            self.class_prompt_definitions,
            self.prompt_config,
        ) = self._load_prompt_config()
        self.prompt_hash = self._build_prompt_hash()
        self.processor = self._load_processor()
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            providers=["CPUExecutionProvider"],
        )
        self.embedding_output_index = self._find_embedding_output_index()
        (
            self.labels,
            self.text_embeddings,
            self.class_groups,
            self.group_labels,
            self.group_embeddings,
        ) = self._load_or_build_text_embeddings()
        self.group_index_by_label = {
            label: index for index, label in enumerate(self.group_labels)
        }
        self.group_to_class_indices = self._build_group_to_class_indices()
        self.gallery = CorrectionGallery(self.gallery_path)

    def learn_correction(self, image_path: str | Path, label: str) -> None:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        original = image.convert("RGB")
        cropped = self._crop_foreground(original)

        original_embedding = self._embed_image(original)
        cropped_embedding = self._embed_image(cropped)
        merged_embedding = _normalize_vector((original_embedding + cropped_embedding) / 2.0)
        
        self.gallery.add(merged_embedding, label)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _load_prompt_config(
        self,
    ) -> tuple[
        dict[str, list[str]],
        list[ClassPromptDefinition],
        dict[str, Any],
    ]:
        with self.prompt_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not config:
            raise ValueError("Prompt configuration is empty.")

        if "classes" not in config:
            class_definitions = [
                ClassPromptDefinition(
                    label=str(label),
                    group="default",
                    prompts=list(prompts),
                )
                for label, prompts in config.items()
            ]
            normalized = {
                "groups": {},
                "classes": {
                    definition.label: {
                        "group": definition.group,
                        "prompts": definition.prompts,
                    }
                    for definition in class_definitions
                },
            }
            return {}, class_definitions, normalized

        group_prompt_config = {
            str(group): list(prompts)
            for group, prompts in config.get("groups", {}).items()
        }
        class_definitions: list[ClassPromptDefinition] = []
        for label, entry in config["classes"].items():
            if isinstance(entry, list):
                prompts = list(entry)
                group = "default"
            else:
                prompts = list(entry["prompts"])
                group = str(entry.get("group", "default"))
            class_definitions.append(
                ClassPromptDefinition(
                    label=str(label),
                    group=group,
                    prompts=prompts,
                )
            )

        normalized = {
            "groups": group_prompt_config,
            "classes": {
                definition.label: {
                    "group": definition.group,
                    "prompts": definition.prompts,
                }
                for definition in class_definitions
            },
        }
        return group_prompt_config, class_definitions, normalized

    def _build_prompt_hash(self) -> str:
        canonical = json.dumps(
            {
                "prompt_config": self.prompt_config,
                "prompt_expansion_version": PROMPT_EXPANSION_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load_processor(self):
        try:
            return AutoProcessor.from_pretrained(
                self.text_model_id,
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001
            return AutoProcessor.from_pretrained(self.text_model_id)

    def _load_text_model(self):
        try:
            return AutoModel.from_pretrained(
                self.text_model_id,
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001
            return AutoModel.from_pretrained(self.text_model_id)

    def _find_embedding_output_index(self) -> int:
        outputs = self.session.get_outputs()
        for index, output in enumerate(outputs):
            shape = output.shape
            if len(shape) == 2 and shape[-1] == 768:
                return index
        return len(outputs) - 1

    def _load_or_build_text_embeddings(
        self,
    ) -> tuple[list[str], np.ndarray, list[str], list[str], np.ndarray]:
        if self.embedding_cache_path.exists():
            with np.load(self.embedding_cache_path, allow_pickle=True) as cache:
                cached_model_id = str(cache["text_model_id"][0])
                cached_prompt_hash = str(cache["prompt_hash"][0])
                if (
                    cached_model_id == self.text_model_id
                    and cached_prompt_hash == self.prompt_hash
                ):
                    labels = [str(item) for item in cache["labels"].tolist()]
                    embeddings = cache["embeddings"].astype(np.float32)
                    if "class_groups" in cache:
                        class_groups = [
                            str(item) for item in cache["class_groups"].tolist()
                        ]
                    else:
                        class_groups = []
                    if "group_labels" in cache:
                        group_labels = [
                            str(item) for item in cache["group_labels"].tolist()
                        ]
                    else:
                        group_labels = []
                    if "group_embeddings" in cache:
                        group_embeddings = cache["group_embeddings"].astype(np.float32)
                    else:
                        group_embeddings = np.empty((0, embeddings.shape[-1]), dtype=np.float32)
                    if not class_groups:
                        class_groups = ["default"] * len(labels)
                    return labels, embeddings, class_groups, group_labels, group_embeddings

        labels: list[str] = []
        embeddings: list[np.ndarray] = []
        class_groups: list[str] = []
        group_labels: list[str] = []
        group_embeddings_list: list[np.ndarray] = []
        text_model = self._load_text_model()
        text_model.eval()

        with torch.no_grad():
            for definition in self.class_prompt_definitions:
                averaged = self._encode_prompt_set(text_model, definition.prompts)
                labels.append(definition.label)
                class_groups.append(definition.group)
                embeddings.append(averaged)

            for group_label, prompts in self.group_prompt_config.items():
                averaged = self._encode_prompt_set(text_model, prompts)
                group_labels.append(group_label)
                group_embeddings_list.append(averaged)

        text_embeddings = np.stack(embeddings, axis=0)
        if group_embeddings_list:
            group_embeddings = np.stack(group_embeddings_list, axis=0)
        else:
            group_embeddings = np.empty((0, text_embeddings.shape[-1]), dtype=np.float32)
        self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.embedding_cache_path,
            labels=np.array(labels),
            embeddings=text_embeddings,
            class_groups=np.array(class_groups),
            group_labels=np.array(group_labels),
            group_embeddings=group_embeddings,
            text_model_id=np.array([self.text_model_id]),
            prompt_hash=np.array([self.prompt_hash]),
        )
        return labels, text_embeddings, class_groups, group_labels, group_embeddings

    def _encode_prompt_set(self, text_model, prompts: list[str]) -> np.ndarray:
        expanded_prompts = self._expand_product_prompts(prompts)
        inputs = self.processor(
            text=expanded_prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_output = text_model.get_text_features(**inputs)
        pooled = self._extract_pooled_output(text_output)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        averaged = pooled.mean(dim=0)
        averaged = averaged / averaged.norm()
        return averaged.cpu().numpy().astype(np.float32)

    def _expand_product_prompts(self, prompts: list[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()
        for prompt in prompts:
            base_prompt = " ".join(str(prompt).split())
            if not base_prompt:
                continue
            for context in PRODUCT_CONTEXT_VARIANTS:
                candidate = base_prompt if not context else f"{base_prompt}, {context}"
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        if not expanded:
            raise ValueError("Each class or group must define at least one non-empty prompt.")
        return expanded

    def _extract_pooled_output(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, "pooler_output"):
            return model_output.pooler_output
        if isinstance(model_output, (list, tuple)) and model_output:
            first_item = model_output[0]
            if isinstance(first_item, torch.Tensor):
                return first_item
        raise TypeError("Unable to extract text embedding tensor from model output.")

    def _build_group_to_class_indices(self) -> dict[str, list[int]]:
        group_to_class_indices: dict[str, list[int]] = {}
        for index, group in enumerate(self.class_groups):
            group_to_class_indices.setdefault(group, []).append(index)
        return group_to_class_indices

    def _crop_foreground(self, image: Image.Image) -> Image.Image:
        rgb_image = image.convert("RGB")
        pixel_array = np.array(rgb_image)
        mask = np.any(pixel_array < WHITE_THRESHOLD, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return rgb_image

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        height, width = pixel_array.shape[:2]
        pad_y = max(8, int((y1 - y0) * FOREGROUND_PADDING_RATIO))
        pad_x = max(8, int((x1 - x0) * FOREGROUND_PADDING_RATIO))

        y0 = max(0, y0 - pad_y)
        x0 = max(0, x0 - pad_x)
        y1 = min(height, y1 + pad_y)
        x1 = min(width, x1 + pad_x)

        cropped = rgb_image.crop((x0, y0, x1, y1))
        side = max(cropped.width, cropped.height)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
        square.paste(cropped, offset)
        return square

    def _check_background_empty(self, image: Image.Image) -> bool:
        """Quick check: if most pixels are near-white, the frame is empty (no object)."""
        arr = np.array(image.convert("RGB"))
        white_pixels = np.all(arr > WHITE_THRESHOLD - 10, axis=2).sum()
        return white_pixels / max(1, arr.shape[0] * arr.shape[1]) > 0.85

    def _check_negative_prompts(self, image_embedding: np.ndarray) -> float:
        """Check if the image matches negative prompts (non-jewelry items)."""
        return self._group_similarity(image_embedding, "negative")

    def _group_similarity(self, image_embedding: np.ndarray, group_label: str) -> float:
        group_index = self.group_index_by_label.get(group_label)
        if group_index is None or group_index >= len(self.group_embeddings):
            return 0.0
        similarity = image_embedding.flatten() @ self.group_embeddings[group_index].T
        return float(similarity)

    def _gold_verification_scores(
        self,
        image_embedding: np.ndarray,
    ) -> tuple[float, float, float]:
        return (
            self._group_similarity(image_embedding, "gold_verification"),
            self._group_similarity(image_embedding, "non_gold_metal"),
            self._check_negative_prompts(image_embedding),
        )

    def _verify_gold_embedding(
        self,
        original: Image.Image,
        image_embedding: np.ndarray,
    ) -> tuple[bool, float, str]:
        if self._check_background_empty(original):
            return False, 0.0, "empty background"

        gold_sim, non_gold_sim, negative_sim = self._gold_verification_scores(
            image_embedding
        )
        rejection_sim = max(non_gold_sim, negative_sim)

        if gold_sim >= rejection_sim and gold_sim > 0.05:
            return (
                True,
                float(gold_sim),
                f"gold (g={gold_sim:.3f} >= reject={rejection_sim:.3f})",
            )
        if gold_sim > 0.12:
            return True, float(gold_sim), f"gold (g={gold_sim:.3f} > 0.12)"
        return (
            False,
            float(rejection_sim),
            (
                "not gold "
                f"(g={gold_sim:.3f}, ng={non_gold_sim:.3f}, neg={negative_sim:.3f})"
            ),
        )

    def _gold_type_has_priority(
        self,
        top_type_similarity: float,
        image_embedding: np.ndarray,
    ) -> tuple[bool, str]:
        gold_sim, non_gold_sim, negative_sim = self._gold_verification_scores(
            image_embedding
        )
        rejection_sim = max(non_gold_sim, negative_sim)
        accepted = (
            top_type_similarity >= GOLD_TYPE_MIN_SIMILARITY
            and top_type_similarity + GOLD_TYPE_REJECTION_MARGIN >= rejection_sim
        )
        reason = (
            f"type={top_type_similarity:.3f}, gold={gold_sim:.3f}, "
            f"non_gold={non_gold_sim:.3f}, negative={negative_sim:.3f}"
        )
        return accepted, reason

    def verify_gold_jewelry(self, image: Image.Image) -> tuple[bool, float, str]:
        """Check whether an image has stronger gold than rejection evidence."""
        original = image.convert("RGB")
        cropped = self._crop_foreground(original)

        merged = _normalize_vector(
            np.expand_dims((self._embed_image(original) + self._embed_image(cropped)) / 2.0, axis=0)
        )[0]
        return self._verify_gold_embedding(original, merged)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        model_inputs = self.processor(images=image, return_tensors="np")
        output_tensors = self.session.run(
            None,
            {self.session.get_inputs()[0].name: model_inputs["pixel_values"].astype(np.float32)},
        )
        embedding = output_tensors[self.embedding_output_index].astype(np.float32)
        return _normalize_vector(embedding)[0]

    def classify_image(
        self,
        image: Image.Image,
        image_path: str | None = None,
    ) -> PredictionResult:
        original = image.convert("RGB")
        cropped = self._crop_foreground(original)

        original_embedding = self._embed_image(original)
        cropped_embedding = self._embed_image(cropped)
        merged_embedding = _normalize_vector(
            np.expand_dims((original_embedding + cropped_embedding) / 2.0, axis=0)
        )[0]

        # Search the correction gallery for a match
        gallery_label, gallery_sim = self.gallery.search(merged_embedding)
        gallery_match = gallery_label is not None
        similarities = merged_embedding @ self.text_embeddings.T
        probabilities = self._compute_class_probabilities(
            similarities,
            merged_embedding,
        )
        scores = [
            ScoreEntry(
                label=label,
                similarity=float(similarities[index]),
                confidence=float(probabilities[index]),
            )
            for index, label in enumerate(self.labels)
        ]
        scores.sort(key=lambda item: item.confidence, reverse=True)
        top_type_similarity = scores[0].similarity

        # Score gold jewel types before applying the non-gold fallback.
        # A close manually learned non-gold match remains authoritative.
        if gallery_match and gallery_label == "Not Gold Jewelry":
            gold_reason = f"gallery match: Not Gold Jewelry (sim={gallery_sim:.3f})"
            return PredictionResult(
                label="Not Gold Jewelry",
                confidence=max(0.99, gallery_sim),
                scores=scores,
                image_path=image_path,
                original_image=original,
                cropped_image=cropped,
                gallery_match=True,
                gallery_similarity=gallery_sim,
                is_gold_jewelry=False,
                gold_verification_reason=gold_reason,
            )

        is_gold, gold_conf, gold_reason = self._verify_gold_embedding(
            original,
            merged_embedding,
        )
        type_has_priority, type_reason = self._gold_type_has_priority(
            top_type_similarity,
            merged_embedding,
        )
        type_has_priority = bool(
            type_has_priority and not self._check_background_empty(original)
        )
        if not is_gold and not type_has_priority:
            # A gallery match for a gold class overrides gold verification —
            # the user manually corrected a similar item before, so trust that.
            if gallery_match and gallery_label != "Not Gold Jewelry":
                is_gold = True
                gold_conf = max(float(gold_conf), gallery_sim)
                gold_reason = f"gallery override: {gallery_label} (sim={gallery_sim:.3f})"
            else:
                return PredictionResult(
                    label="Not Gold Jewelry",
                    confidence=float(gold_conf),
                    scores=scores,
                    image_path=image_path,
                    original_image=original,
                    cropped_image=cropped,
                    gallery_match=gallery_match,
                    gallery_similarity=gallery_sim,
                    is_gold_jewelry=False,
                    gold_verification_reason=gold_reason,
                )

        final_label = gallery_label if gallery_match else scores[0].label
        # If gallery match, boost the top score confidence or just set to high value
        final_confidence = max(0.99, scores[0].confidence) if gallery_match else scores[0].confidence

        return PredictionResult(
            label=final_label,
            confidence=final_confidence,
            scores=scores,
            image_path=image_path,
            original_image=original,
            cropped_image=cropped,
            gallery_match=gallery_match,
            gallery_similarity=gallery_sim,
            is_gold_jewelry=True,
            gold_verification_reason=(
                gold_reason
                if is_gold or gallery_match
                else (
                    f"gold type priority: {scores[0].label} "
                    f"(similarity={scores[0].similarity:.3f}; {type_reason})"
                )
            ),
        )

    def _compute_class_probabilities(
        self,
        class_similarities: np.ndarray,
        image_embedding: np.ndarray,
    ) -> np.ndarray:
        if not self.group_labels:
            scaled = class_similarities * SIMILARITY_SCALE
            return torch.softmax(torch.from_numpy(scaled), dim=-1).numpy()

        group_similarities = image_embedding @ self.group_embeddings.T
        group_scaled = group_similarities * SIMILARITY_SCALE
        group_probabilities = torch.softmax(torch.from_numpy(group_scaled), dim=-1).numpy()

        class_probabilities = np.zeros_like(class_similarities, dtype=np.float32)
        for group_label, class_indices in self.group_to_class_indices.items():
            group_index = self.group_index_by_label.get(group_label)
            if group_index is None:
                continue
            group_probability = float(group_probabilities[group_index])
            group_scores = class_similarities[class_indices] * SIMILARITY_SCALE
            within_group = torch.softmax(torch.from_numpy(group_scores), dim=-1).numpy()
            for class_index, class_probability in zip(class_indices, within_group, strict=False):
                class_probabilities[class_index] = group_probability * float(
                    class_probability
                )

        total = float(class_probabilities.sum())
        if total <= 0:
            scaled = class_similarities * SIMILARITY_SCALE
            return torch.softmax(torch.from_numpy(scaled), dim=-1).numpy()
        return class_probabilities / total

    def classify_path(self, image_path: str | Path) -> PredictionResult:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        return self.classify_image(image, image_path=str(path))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot jewelry classifier using SigLIP2 ONNX image encoder.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ONNX_MODEL,
        help="Path to the ONNX vision encoder model.",
    )
    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPT_FILE,
        help="Path to the prompt JSON file.",
    )
    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL_ID,
        help="Hugging Face text model id used to build label embeddings.",
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE_FILE,
        help="Path to the cached text embeddings file.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    classifier = JewelryZeroShotClassifier(
        onnx_model_path=args.model,
        prompt_path=args.prompts,
        text_model_id=args.text_model,
        embedding_cache_path=args.cache,
    )
    prediction = classifier.classify_path(args.image)

    print(f"Image: {prediction.image_path}")
    print(f"Predicted class: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print("Scores:")
    for score in prediction.scores:
        print(
            f"  - {score.label:<14} confidence={score.confidence:.2%} "
            f"similarity={score.similarity:.4f}"
        )


if __name__ == "__main__":
    main()
