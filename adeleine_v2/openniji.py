from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional
from urllib.parse import urlparse

import cv2 as cv
import numpy as np
import requests
from torch.utils.data import Dataset

from .atari import AtariHintGenerator
from .conditions import ColorizationCondition, ModalityDropout
from .lineart import LineArtAugmentor, LineArtPaths, available_methods
from .reference import ReferenceSelector
from .reference_conditioning import ReferenceConditioningBuilder, ReferenceConditioningConfig, split_reference_layers


ReferencePolicy = Literal["self", "deformed_self", "self_deformed", "sibling", "mixed", "none"]
DEFAULT_REFERENCE_POLICY: ReferencePolicy = "deformed_self"


def resolve_reference_policy(policy: ReferencePolicy) -> ReferencePolicy:
    if policy == "mixed":
        return "deformed_self"
    return policy


# Masks are warped with the same border mode as the RGB (BORDER_REFLECT_101): reflected foreground pulled in
# from the image border must stay labelled foreground, or it leaks into the background layer.
def _warp_mask(mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int], perspective: bool = False) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if perspective:
        out = cv.warpPerspective(mask_arr, matrix, size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    else:
        out = cv.warpAffine(mask_arr, matrix, size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    return np.clip(out, 0, 255).astype(np.uint8)[..., None]


ReferenceDeformStrength = Literal["mild", "strong"]
ReferenceBackgroundSource = Literal["self", "other"]

# (max rotation deg, scale range, max translation fraction, perspective jitter range, hflip prob, elastic amplitude fraction)
_DEFORM_PARAMS = {
    "mild": (12.0, (0.86, 1.14), 0.10, (0.015, 0.055), 0.0, 0.0),
    "strong": (25.0, (0.75, 1.25), 0.18, (0.03, 0.09), 0.5, 0.04),
}


def _elastic_maps(h: int, w: int, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    # Smooth random displacement field: coarse noise upsampled to full resolution.
    grid = 5
    dx = cv.resize(np.random.uniform(-1.0, 1.0, (grid, grid)).astype(np.float32), (w, h), interpolation=cv.INTER_CUBIC)
    dy = cv.resize(np.random.uniform(-1.0, 1.0, (grid, grid)).astype(np.float32), (w, h), interpolation=cv.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return xx + dx * amplitude * w, yy + dy * amplitude * h


def deform_reference_rgb_and_mask(
    rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: ReferenceDeformStrength = "mild",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    max_angle, scale_range, max_shift, jitter_range, flip_prob, elastic = _DEFORM_PARAMS[strength]
    h, w = rgb.shape[:2]
    if flip_prob > 0.0 and np.random.random() < flip_prob:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        if mask is not None:
            mask = np.ascontiguousarray(np.asarray(mask)[:, ::-1])
    center = (w * 0.5, h * 0.5)
    angle = float(np.random.uniform(-max_angle, max_angle))
    scale = float(np.random.uniform(*scale_range))
    matrix = cv.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += float(np.random.uniform(-max_shift, max_shift) * w)
    matrix[1, 2] += float(np.random.uniform(-max_shift, max_shift) * h)
    out = cv.warpAffine(rgb, matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    mask_out = _warp_mask(mask, matrix, (w, h)) if mask is not None else None

    if elastic > 0.0:
        map_x, map_y = _elastic_maps(h, w, elastic)
        out = cv.remap(out, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        if mask_out is not None:
            mask_out = cv.remap(mask_out[..., 0], map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)[..., None]

    jitter = min(h, w) * float(np.random.uniform(*jitter_range))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.random.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
    perspective = cv.getPerspectiveTransform(src, dst)
    out = cv.warpPerspective(out, perspective, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    if mask_out is not None:
        mask_out = _warp_mask(mask_out, perspective, (w, h), perspective=True)

    work = out.astype(np.float32)
    work = (work - 127.5) * float(np.random.uniform(0.88, 1.12)) + 127.5
    work += float(np.random.uniform(-10.0, 10.0))
    work += np.random.normal(0.0, float(np.random.uniform(1.0, 4.0)), size=work.shape).astype(np.float32)
    out = np.clip(work, 0, 255).astype(np.uint8)

    if np.random.random() < 0.4:
        sigma = float(np.random.uniform(0.3, 0.8))
        out = cv.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if mask_out is not None:
            mask_out = cv.GaussianBlur(mask_out, (0, 0), sigmaX=sigma, sigmaY=sigma)
            if mask_out.ndim == 2:
                mask_out = mask_out[..., None]

    if np.random.random() < 0.35:
        factor = float(np.random.uniform(0.65, 0.90))
        small_w = max(16, int(w * factor))
        small_h = max(16, int(h * factor))
        small = cv.resize(out, (small_w, small_h), interpolation=cv.INTER_AREA)
        out = cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR)
        if mask_out is not None:
            small_mask = cv.resize(mask_out[..., 0], (small_w, small_h), interpolation=cv.INTER_AREA)
            mask_out = cv.resize(small_mask, (w, h), interpolation=cv.INTER_LINEAR)[..., None]

    if np.random.random() < 0.25:
        mean_color = tuple(int(x) for x in out.reshape(-1, 3).mean(axis=0))
        for _ in range(int(np.random.randint(1, 4))):
            box_w = int(np.random.uniform(0.04, 0.12) * w)
            box_h = int(np.random.uniform(0.04, 0.12) * h)
            x0 = int(np.random.uniform(0, max(1, w - box_w)))
            y0 = int(np.random.uniform(0, max(1, h - box_h)))
            cv.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), mean_color, thickness=-1)

    if mask_out is not None:
        mask_out = np.clip(mask_out, 0, 255).astype(np.uint8)
    return out, mask_out


def deform_reference_rgb(rgb: np.ndarray) -> np.ndarray:
    return deform_reference_rgb_and_mask(rgb, None)[0]


@dataclass
class ReferenceInputs:
    """Per-sample reference images plus optional precomputed conditioning, aligned by index."""

    images: List[np.ndarray] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    policy: str = "none"
    masks: List[Optional[np.ndarray]] = field(default_factory=list)
    tags: List[Optional[List[str]]] = field(default_factory=list)
    wd_indices: List[Optional[List[np.ndarray]]] = field(default_factory=list)
    wd_scores: List[Optional[List[np.ndarray]]] = field(default_factory=list)
    backgrounds: List[Optional[np.ndarray]] = field(default_factory=list)

    def build(self, conditioner: ReferenceConditioningBuilder):
        return conditioner.build(
            self.images,
            self.masks,
            self.tags,
            self.wd_indices,
            self.wd_scores,
            reference_backgrounds=self.backgrounds,
        )


def self_reference_inputs(
    conditioner: ReferenceConditioningBuilder,
    source: np.ndarray,
    url: str,
    policy: str,
    resize: Callable[[np.ndarray], np.ndarray],
    deform_strength: ReferenceDeformStrength = "mild",
    background_source: ReferenceBackgroundSource = "self",
    other_rgb: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> ReferenceInputs:
    """Self-derived references for the self / deformed_self / self_deformed policies.

    Only (image, mask) pairs are handed to the builder, which splits foreground and background exactly as
    it does for user references at inference. The deformed reference therefore never carries a pixel-aligned
    copy of the target: its background is split from the deformed image, or taken from another record.
    """
    mask = conditioner.reference_mask(source) if conditioner.enabled else None
    tags = conditioner.tag(source, mask) if conditioner.wants_tags else []
    wd_indices: Optional[List[np.ndarray]] = None
    wd_scores: Optional[List[np.ndarray]] = None
    if conditioner.wants_wd_tokens:
        # Same foreground/background layers as inference; these are what precompute_wd_reference_tokens caches.
        results = conditioner.wd_results(source, mask)
        wd_indices, wd_scores = [r.indices for r in results], [r.scores for r in results]

    images: List[np.ndarray] = []
    masks: List[Optional[np.ndarray]] = []
    if policy in {"self", "self_deformed"}:
        images.append(source)
        masks.append(mask)
    if policy in {"deformed_self", "self_deformed"}:
        deformed, deformed_mask = deform_reference_rgb_and_mask(source, mask, deform_strength)
        deformed = resize(deformed)
        if deformed_mask is not None:
            deformed_mask = resize(deformed_mask)
            if deformed_mask.ndim == 2:
                deformed_mask = deformed_mask[..., None]
        images.append(deformed)
        masks.append(deformed_mask)

    backgrounds: List[Optional[np.ndarray]] = [None] * len(images)
    if conditioner.enabled and background_source == "other" and other_rgb is not None:
        other = other_rgb()
        if other is not None:
            _, other_bg, _ = split_reference_layers(other, conditioner.reference_mask(other))
            backgrounds = [other_bg] * len(images)

    n = len(images)
    return ReferenceInputs(images, [url] * n, policy, masks, [tags] * n, [wd_indices] * n, [wd_scores] * n, backgrounds)


@dataclass(frozen=True)
class OpenNijiRecord:
    url: str
    prompt: str
    style: str
    image_path: Path
    group_key: str


def default_openniji_jsonl(hf_home: Path = Path("/data/shasegawa/adeleine/huggingface")) -> Path:
    root = hf_home / "hub" / "datasets--ShoukanLabs--OpenNiji-Dataset" / "snapshots"
    candidates = sorted(root.glob("*/dataset.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            "OpenNiji dataset.jsonl was not found. Run: "
            "hf download ShoukanLabs/OpenNiji-Dataset --repo-type dataset "
            f"--cache-dir {hf_home / 'hub'}"
        )
    return candidates[-1]


def prompt_group_key(prompt: str) -> str:
    text = prompt.lower()
    text = re.sub(r"-?\s*image\s*#?\d+", "", text)
    text = re.sub(r"<@!?\d+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def image_cache_path(url: str, image_cache: Path) -> Path:
    suffix = Path(urlparse(url).path).suffix.lower() or ".png"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return image_cache / f"{digest}{suffix}"


def read_records(jsonl_path: Path, image_cache: Path, max_records: Optional[int] = None) -> List[OpenNijiRecord]:
    records: List[OpenNijiRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            url = obj.get("url", "")
            prompt = obj.get("prompt", "")
            style = obj.get("style", "")
            if not url:
                continue
            records.append(
                OpenNijiRecord(
                    url=url,
                    prompt=prompt,
                    style=style,
                    image_path=image_cache_path(url, image_cache),
                    group_key=prompt_group_key(prompt),
                )
            )
    return records


class OpenNijiImageCache:
    def __init__(self, image_cache: Path, timeout: float = 20.0):
        self.image_cache = image_cache
        self.timeout = timeout
        self.image_cache.mkdir(parents=True, exist_ok=True)

    def get(self, record: OpenNijiRecord, download: bool = True) -> Path:
        if record.image_path.exists():
            return record.image_path
        if not download:
            raise FileNotFoundError(record.image_path)
        tmp = record.image_path.with_suffix(record.image_path.suffix + ".tmp")
        response = requests.get(record.url, timeout=self.timeout)
        response.raise_for_status()
        tmp.write_bytes(response.content)
        image = cv.imread(str(tmp), cv.IMREAD_COLOR)
        if image is None:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file is not a readable image: {record.url}")
        tmp.replace(record.image_path)
        return record.image_path


class OpenNijiColorizationDataset(Dataset):
    """Adeleine v2 training dataset backed by ShoukanLabs/OpenNiji-Dataset."""

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        image_cache: Path = Path("/data/shasegawa/adeleine/openniji/images"),
        hf_home: Path = Path("/data/shasegawa/adeleine/huggingface"),
        sketch_root: Optional[Path] = None,
        digital_root: Optional[Path] = None,
        anime_line_root: Optional[Path] = None,
        image_size: int = 512,
        max_records: Optional[int] = None,
        max_refs: int = 4,
        download: bool = True,
        line_methods: tuple[str, ...] = ("xdog", "pencil", "digital", "lineart_anime", "blend"),
        dropout: Optional[ModalityDropout] = None,
        reference_policy: ReferencePolicy = DEFAULT_REFERENCE_POLICY,
        reference_deform_strength: ReferenceDeformStrength = "mild",
        reference_background_source: ReferenceBackgroundSource = "self",
        reference_conditioning: str = "none",
        reference_tag_cache_root: Optional[Path] = None,
        reference_mask_root: Optional[Path] = None,
        reference_mask_fallback: str = "grabcut",
        skytnt_repo: Optional[Path] = None,
        skytnt_model_id: str = "skytnt/anime-seg",
        skytnt_ckpt: Optional[Path] = None,
        skytnt_net: str = "isnet_is",
        skytnt_image_size: int = 1024,
        skytnt_device: str = "cuda:0",
        skytnt_fp32: bool = False,
        skytnt_local_files_only: bool = False,
        reference_cache_generated_masks: bool = True,
        wd_tagger_model: Optional[Path] = None,
        wd_tagger_labels: Optional[Path] = None,
        wd_tagger_threshold: float = 0.35,
        wd_tagger_character_threshold: float = 0.85,
        wd_tagger_max_tokens: int = 32,
        reference_tag_max: int = 24,
    ):
        self.jsonl_path = jsonl_path or default_openniji_jsonl(hf_home)
        self.image_cache = OpenNijiImageCache(image_cache)
        self.records = read_records(self.jsonl_path, image_cache, max_records=max_records)
        self.image_size = image_size
        self.max_refs = max_refs
        self.download = download
        self.reference_policy = reference_policy
        line_paths = LineArtPaths(pencil_dir=sketch_root, digital_dir=digital_root, anime_dir=anime_line_root)
        self.lineart = LineArtAugmentor(available_methods(list(line_methods), line_paths), line_paths)
        self.atari = AtariHintGenerator()
        self.references = ReferenceSelector(max_refs=max_refs, refs_per_patch=max(1, max_refs // 4), image_size=image_size)
        self.reference_conditioner = ReferenceConditioningBuilder(
            ReferenceConditioningConfig(
                mode=reference_conditioning,
                tag_cache_root=reference_tag_cache_root,
                mask_root=reference_mask_root,
                mask_fallback=reference_mask_fallback,
                skytnt_repo=skytnt_repo,
                skytnt_model_id=skytnt_model_id,
                skytnt_ckpt=skytnt_ckpt,
                skytnt_net=skytnt_net,
                skytnt_image_size=skytnt_image_size,
                skytnt_device=skytnt_device,
                skytnt_fp32=skytnt_fp32,
                skytnt_local_files_only=skytnt_local_files_only,
                cache_generated_masks=reference_cache_generated_masks,
                wd_tagger_model=wd_tagger_model,
                wd_tagger_labels=wd_tagger_labels,
                wd_tagger_threshold=wd_tagger_threshold,
                wd_tagger_character_threshold=wd_tagger_character_threshold,
                wd_tagger_max_tokens=wd_tagger_max_tokens,
                max_tags=reference_tag_max,
            )
        )
        self.dropout = dropout or ModalityDropout()
        self.reference_deform_strength = reference_deform_strength
        self.reference_background_source = reference_background_source
        self.groups = self._build_groups(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ColorizationCondition:
        record = self.records[index]
        image_path = self.image_cache.get(record, download=self.download)
        color_bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(image_path)
        color_bgr = self._resize_square(color_bgr)
        color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

        line_bgr = self.lineart(image_path, color_bgr=color_bgr)
        line_bgr = self._resize_square(line_bgr)
        line_rgb = cv.cvtColor(line_bgr, cv.COLOR_BGR2RGB)
        atari_rgb, atari_mask = self.atari(color_rgb, line_rgb)

        refs = self._reference_images(record, line_rgb, color_rgb)
        ref_cond = refs.build(self.reference_conditioner)
        text = self._caption(record)
        condition = ColorizationCondition(
            lineart=line_rgb,
            target=color_rgb,
            atari_rgb=atari_rgb,
            atari_mask=atari_mask,
            references=refs.images,
            reference_foregrounds=ref_cond.foregrounds,
            reference_backgrounds=ref_cond.backgrounds,
            reference_masks=ref_cond.masks,
            reference_tags=ref_cond.tags,
            reference_wd_indices=ref_cond.wd_indices,
            reference_wd_scores=ref_cond.wd_scores,
            text=text,
            metadata={"url": record.url, "prompt": record.prompt, "style": record.style, "image_path": str(image_path), "reference_urls": refs.urls, "reference_policy": self.reference_policy, "reference_policy_actual": refs.policy, "reference_tags": ref_cond.tags},
        )
        return self.dropout(condition)

    def _reference_images(self, record: OpenNijiRecord, line_rgb: np.ndarray, color_rgb: np.ndarray) -> ReferenceInputs:
        policy = resolve_reference_policy(self.reference_policy)
        if policy == "none":
            return ReferenceInputs(policy=policy)
        if policy in {"self", "deformed_self", "self_deformed"}:
            return self_reference_inputs(
                self.reference_conditioner,
                self._resize_square(color_rgb),
                record.url,
                policy,
                self._resize_square,
                self.reference_deform_strength,
                self.reference_background_source,
                lambda: self._random_other_rgb(record),
            )

        siblings = [item for item in self.groups.get(record.group_key, []) if item.url != record.url]
        if not siblings:
            return ReferenceInputs(policy=policy)
        np.random.shuffle(siblings)
        refs = []
        ref_urls = []
        for sibling in siblings[: self.max_refs]:
            try:
                path = self.image_cache.get(sibling, download=self.download)
                img = cv.imread(str(path), cv.IMREAD_COLOR)
                if img is None:
                    continue
                refs.append(cv.cvtColor(self._resize_square(img), cv.COLOR_BGR2RGB))
                ref_urls.append(sibling.url)
            except Exception:
                continue
        pack = self.references.pack(line_rgb, refs)
        return ReferenceInputs(pack.images, ref_urls[: len(pack.images)], policy)

    def _resize_square(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, (self.image_size, self.image_size), interpolation=cv.INTER_AREA)

    def _random_other_rgb(self, record: OpenNijiRecord, attempts: int = 3) -> Optional[np.ndarray]:
        for _ in range(attempts):
            other = self.records[np.random.randint(len(self.records))]
            if other.url == record.url:
                continue
            try:
                img = cv.imread(str(self.image_cache.get(other, download=self.download)), cv.IMREAD_COLOR)
            except Exception:
                continue
            if img is not None:
                return cv.cvtColor(self._resize_square(img), cv.COLOR_BGR2RGB)
        return None

    @staticmethod
    def _caption(record: OpenNijiRecord) -> str:
        return f"{record.prompt}, style: {record.style}" if record.style else record.prompt

    @staticmethod
    def _build_groups(records: List[OpenNijiRecord]) -> Dict[str, List[OpenNijiRecord]]:
        groups: Dict[str, List[OpenNijiRecord]] = {}
        for record in records:
            groups.setdefault(record.group_key, []).append(record)
        return groups


@dataclass(frozen=True)
class OpenNijiParquetRecord:
    parquet_path: Path
    row_group: int
    row_in_group: int
    prompt: str
    style: str
    url: str
    group_key: str


def _openniji_repo_start(repo_id_or_safe: str) -> int:
    name = repo_id_or_safe.rsplit("/", 1)[-1].replace("datasets--ShoukanLabs--", "")
    match = re.match(r"OpenNiji-(\d+)_", name)
    return int(match.group(1)) if match else 10**12


def default_openniji_parquet_roots(
    repo_id: str = "ShoukanLabs/OpenNiji-0_32237",
    hf_home: Path = Path("/data/shasegawa/adeleine/huggingface"),
) -> List[Path]:
    if repo_id != "all":
        return [default_openniji_parquet_root(repo_id, hf_home)]

    root = hf_home / "hub"
    candidates = []
    for repo_root in root.glob("datasets--ShoukanLabs--OpenNiji-*"):
        if repo_root.name == "datasets--ShoukanLabs--OpenNiji-Dataset":
            continue
        snapshots = sorted((repo_root / "snapshots").glob("*"))
        if snapshots:
            candidates.append((_openniji_repo_start(repo_root.name), snapshots[-1]))
    if not candidates:
        raise FileNotFoundError(
            "No OpenNiji split repositories were found. Download them with "
            "hf download ShoukanLabs/OpenNiji-*_... or set a single --openniji_repo_id."
        )
    return [path for _, path in sorted(candidates, key=lambda item: item[0])]


def default_openniji_parquet_root(
    repo_id: str = "ShoukanLabs/OpenNiji-0_32237",
    hf_home: Path = Path("/data/shasegawa/adeleine/huggingface"),
) -> Path:
    safe = "datasets--" + repo_id.replace("/", "--")
    root = hf_home / "hub" / safe / "snapshots"
    snapshots = sorted(root.glob("*"))
    if not snapshots:
        raise FileNotFoundError(
            f"OpenNiji Parquet split {repo_id} was not found. Run: "
            f"hf download {repo_id} --repo-type dataset --cache-dir {hf_home / 'hub'}"
        )
    return snapshots[-1]


def parquet_files(parquet_root: Path, pattern: str = "data/*.parquet") -> List[Path]:
    files = sorted(parquet_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No OpenNiji parquet files found under {parquet_root} with {pattern}")
    return files


def parquet_files_from_roots(parquet_roots: List[Path], pattern: str = "data/*.parquet") -> List[Path]:
    files: List[Path] = []
    for root in parquet_roots:
        files.extend(parquet_files(root, pattern))
    if not files:
        raise FileNotFoundError(f"No OpenNiji parquet files found under {parquet_roots} with {pattern}")
    return files


class OpenNijiParquetColorizationDataset(Dataset):
    """Adeleine v2 dataset for the image-backed OpenNiji split repos."""

    def __init__(
        self,
        parquet_root: Optional[Path] = None,
        repo_id: str = "ShoukanLabs/OpenNiji-0_32237",
        hf_home: Path = Path("/data/shasegawa/adeleine/huggingface"),
        parquet_pattern: str = "data/*.parquet",
        sketch_root: Optional[Path] = None,
        digital_root: Optional[Path] = None,
        anime_line_root: Optional[Path] = None,
        image_size: int = 512,
        max_records: Optional[int] = None,
        max_refs: int = 4,
        line_methods: tuple[str, ...] = ("xdog", "pencil", "digital", "lineart_anime", "blend"),
        dropout: Optional[ModalityDropout] = None,
        reference_policy: ReferencePolicy = DEFAULT_REFERENCE_POLICY,
        reference_deform_strength: ReferenceDeformStrength = "mild",
        reference_background_source: ReferenceBackgroundSource = "self",
        reference_conditioning: str = "none",
        reference_tag_cache_root: Optional[Path] = None,
        reference_mask_root: Optional[Path] = None,
        reference_mask_fallback: str = "grabcut",
        skytnt_repo: Optional[Path] = None,
        skytnt_model_id: str = "skytnt/anime-seg",
        skytnt_ckpt: Optional[Path] = None,
        skytnt_net: str = "isnet_is",
        skytnt_image_size: int = 1024,
        skytnt_device: str = "cuda:0",
        skytnt_fp32: bool = False,
        skytnt_local_files_only: bool = False,
        reference_cache_generated_masks: bool = True,
        wd_tagger_model: Optional[Path] = None,
        wd_tagger_labels: Optional[Path] = None,
        wd_tagger_threshold: float = 0.35,
        wd_tagger_character_threshold: float = 0.85,
        wd_tagger_max_tokens: int = 32,
        reference_tag_max: int = 24,
    ):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Install pyarrow to read OpenNiji Parquet splits") from exc

        self.pq = pq
        if parquet_root is not None:
            self.parquet_roots = [parquet_root]
        else:
            self.parquet_roots = default_openniji_parquet_roots(repo_id, hf_home)
        self.parquet_root = self.parquet_roots[0]
        self.parquet_paths = parquet_files_from_roots(self.parquet_roots, parquet_pattern)
        self.image_size = image_size
        self.max_refs = max_refs
        self.reference_policy = reference_policy
        line_paths = LineArtPaths(pencil_dir=sketch_root, digital_dir=digital_root, anime_dir=anime_line_root)
        self.lineart = LineArtAugmentor(available_methods(list(line_methods), line_paths), line_paths)
        self.atari = AtariHintGenerator()
        self.references = ReferenceSelector(max_refs=max_refs, refs_per_patch=max(1, max_refs // 4), image_size=image_size)
        self.reference_conditioner = ReferenceConditioningBuilder(
            ReferenceConditioningConfig(
                mode=reference_conditioning,
                tag_cache_root=reference_tag_cache_root,
                mask_root=reference_mask_root,
                mask_fallback=reference_mask_fallback,
                skytnt_repo=skytnt_repo,
                skytnt_model_id=skytnt_model_id,
                skytnt_ckpt=skytnt_ckpt,
                skytnt_net=skytnt_net,
                skytnt_image_size=skytnt_image_size,
                skytnt_device=skytnt_device,
                skytnt_fp32=skytnt_fp32,
                skytnt_local_files_only=skytnt_local_files_only,
                cache_generated_masks=reference_cache_generated_masks,
                wd_tagger_model=wd_tagger_model,
                wd_tagger_labels=wd_tagger_labels,
                wd_tagger_threshold=wd_tagger_threshold,
                wd_tagger_character_threshold=wd_tagger_character_threshold,
                wd_tagger_max_tokens=wd_tagger_max_tokens,
                max_tags=reference_tag_max,
            )
        )
        self.dropout = dropout or ModalityDropout()
        self.reference_deform_strength = reference_deform_strength
        self.reference_background_source = reference_background_source
        self.records = self._build_records(max_records)
        self.groups = self._build_groups(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ColorizationCondition:
        last_exc: Optional[Exception] = None
        max_attempts = min(len(self.records), 32)
        for offset in range(max_attempts):
            record = self.records[(index + offset) % len(self.records)]
            try:
                return self._getitem_record(record)
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Could not decode a valid OpenNiji image after {max_attempts} attempts") from last_exc

    def _getitem_record(self, record: OpenNijiParquetRecord) -> ColorizationCondition:
        row = self._read_record(record)
        color_bgr = self._decode_image(row["image"]["bytes"])
        color_bgr = self._resize_square(color_bgr)
        color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

        pseudo_path = self._line_cache_name(record)
        line_bgr = self.lineart(pseudo_path, color_bgr=color_bgr)
        line_bgr = self._resize_square(line_bgr)
        line_rgb = cv.cvtColor(line_bgr, cv.COLOR_BGR2RGB)
        atari_rgb, atari_mask = self.atari(color_rgb, line_rgb)

        refs = self._reference_images(record, line_rgb, color_rgb)
        ref_cond = refs.build(self.reference_conditioner)
        condition = ColorizationCondition(
            lineart=line_rgb,
            target=color_rgb,
            atari_rgb=atari_rgb,
            atari_mask=atari_mask,
            references=refs.images,
            reference_foregrounds=ref_cond.foregrounds,
            reference_backgrounds=ref_cond.backgrounds,
            reference_masks=ref_cond.masks,
            reference_tags=ref_cond.tags,
            reference_wd_indices=ref_cond.wd_indices,
            reference_wd_scores=ref_cond.wd_scores,
            text=self._caption(record),
            metadata={"url": record.url, "prompt": record.prompt, "style": record.style, "parquet_path": str(record.parquet_path), "reference_urls": refs.urls, "reference_policy": self.reference_policy, "reference_policy_actual": refs.policy, "reference_tags": ref_cond.tags},
        )
        return self.dropout(condition)

    def _build_records(self, max_records: Optional[int]) -> List[OpenNijiParquetRecord]:
        records: List[OpenNijiParquetRecord] = []
        for path in self.parquet_paths:
            pf = self.pq.ParquetFile(path)
            for row_group in range(pf.num_row_groups):
                table = pf.read_row_group(row_group, columns=["url", "prompt", "style"])
                for row_in_group, row in enumerate(table.to_pylist()):
                    records.append(
                        OpenNijiParquetRecord(
                            parquet_path=path,
                            row_group=row_group,
                            row_in_group=row_in_group,
                            prompt=row.get("prompt") or "",
                            style=row.get("style") or "",
                            url=row.get("url") or "",
                            group_key=prompt_group_key(row.get("prompt") or ""),
                        )
                    )
                    if max_records is not None and len(records) >= max_records:
                        return records
        return records

    def _read_record(self, record: OpenNijiParquetRecord) -> dict:
        table = self.pq.ParquetFile(record.parquet_path).read_row_group(record.row_group)
        return table.slice(record.row_in_group, 1).to_pylist()[0]

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("OpenNiji image bytes are empty")
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        if data.size == 0:
            raise ValueError("OpenNiji image byte buffer is empty")
        try:
            image = cv.imdecode(data, cv.IMREAD_COLOR)
        except cv.error as exc:
            raise ValueError("OpenNiji image bytes could not be decoded by OpenCV") from exc
        if image is None:
            raise ValueError("OpenNiji image bytes could not be decoded")
        return image


    @staticmethod
    def _line_cache_name(record: OpenNijiParquetRecord) -> Path:
        digest = hashlib.sha256(record.url.encode("utf-8")).hexdigest()
        return Path(f"{digest}.png")

    def _reference_images(self, record: OpenNijiParquetRecord, line_rgb: np.ndarray, color_rgb: np.ndarray) -> ReferenceInputs:
        policy = resolve_reference_policy(self.reference_policy)
        if policy == "none":
            return ReferenceInputs(policy=policy)
        if policy in {"self", "deformed_self", "self_deformed"}:
            return self_reference_inputs(
                self.reference_conditioner,
                self._resize_square(color_rgb),
                record.url,
                policy,
                self._resize_square,
                self.reference_deform_strength,
                self.reference_background_source,
                lambda: self._random_other_rgb(record),
            )

        siblings = [item for item in self.groups.get(record.group_key, []) if item != record]
        if not siblings:
            return ReferenceInputs(policy=policy)
        np.random.shuffle(siblings)
        refs = []
        ref_urls = []
        for sibling in siblings[: self.max_refs]:
            try:
                row = self._read_record(sibling)
                img = self._decode_image(row["image"]["bytes"])
                refs.append(cv.cvtColor(self._resize_square(img), cv.COLOR_BGR2RGB))
                ref_urls.append(sibling.url)
            except Exception:
                continue
        pack = self.references.pack(line_rgb, refs)
        return ReferenceInputs(pack.images, ref_urls[: len(pack.images)], policy)

    def _resize_square(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, (self.image_size, self.image_size), interpolation=cv.INTER_AREA)

    def _random_other_rgb(self, record: OpenNijiParquetRecord, attempts: int = 3) -> Optional[np.ndarray]:
        for _ in range(attempts):
            other = self.records[np.random.randint(len(self.records))]
            if other == record:
                continue
            try:
                img = self._decode_image(self._read_record(other)["image"]["bytes"])
            except Exception:
                continue
            return cv.cvtColor(self._resize_square(img), cv.COLOR_BGR2RGB)
        return None

    @staticmethod
    def _caption(record: OpenNijiParquetRecord) -> str:
        return f"{record.prompt}, style: {record.style}" if record.style else record.prompt

    @staticmethod
    def _build_groups(records: List[OpenNijiParquetRecord]) -> Dict[str, List[OpenNijiParquetRecord]]:
        groups: Dict[str, List[OpenNijiParquetRecord]] = {}
        for record in records:
            groups.setdefault(record.group_key, []).append(record)
        return groups


def prefetch_openniji_images(
    jsonl_path: Optional[Path] = None,
    image_cache: Path = Path("/data/shasegawa/adeleine/openniji/images"),
    hf_home: Path = Path("/data/shasegawa/adeleine/huggingface"),
    max_records: Optional[int] = None,
) -> tuple[int, int]:
    records = read_records(jsonl_path or default_openniji_jsonl(hf_home), image_cache, max_records=max_records)
    cache = OpenNijiImageCache(image_cache)
    ok = 0
    failed = 0
    for record in records:
        try:
            cache.get(record, download=True)
            ok += 1
        except Exception:
            failed += 1
    return ok, failed
