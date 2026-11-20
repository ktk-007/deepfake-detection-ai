"""The Unmask analysis pipeline (PLAN 12): video file -> 32 frames -> face -> crops -> scores -> verdict.

Models are found at start-up and on each request: weights\\<key>.pt first (the tracked final checkpoints),
then the queue's runs\\<run>\\best.pt (avg.pt when averaging was adopted). Calibration (one temperature and the
two band thresholds per model, fitted on part 08 in Phase 4) is read from results\\calibration.json; until it
exists the plan's fallback band 0.35 / 0.65 and temperature 1 are used, and the result says so.
Nothing here is shown that the model or results\\ did not produce.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..common import PROJECT_ROOT
from ..data.dfdc_zip import read_frames_file
from ..data.faces import crops_for

RUNS = PROJECT_ROOT / "runs"
WEIGHTS = PROJECT_ROOT / "weights"
RESULTS = PROJECT_ROOT / "results"
N_FRAMES = 32
N_HEATMAPS = 6
FALLBACK_BAND = (0.35, 0.65)
STEPS = ("extract", "detect", "analyze", "result")

# Dropdown order (PLAN 12): final ViT-L/14, ViT-B/16 (faster), EfficientNet-B4, then the ensemble if adopted.
CANDIDATES = [
    ("vit_l14", "CLIP ViT-L/14", ["e7_l14"]),
    ("vit_b16", "CLIP ViT-B/16 (faster)", ["e2_b16", "a7b_ce_lr3e-4"]),
    ("effb4", "EfficientNet-B4", ["e3_effb4"]),
]


class AnalysisError(Exception):
    """Shown to the user: kind is 'no_face' or 'unreadable'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class ModelEntry:
    key: str
    label: str
    path: Path | None = None
    members: list = field(default_factory=list)   # ensemble only

    def public(self) -> dict:
        return {"key": self.key, "label": self.label}


def _ckpt_for_run(run_dir: Path) -> Path | None:
    dec = run_dir / "avg_decision.json"
    if dec.exists():
        try:
            if json.loads(dec.read_text(encoding="utf-8")).get("use_avg") and (run_dir / "avg.pt").exists():
                return run_dir / "avg.pt"
        except (OSError, ValueError):
            pass
    best = run_dir / "best.pt"
    return best if best.exists() else None


def find_models(runs: Path = RUNS, weights: Path = WEIGHTS, results: Path = RESULTS) -> list[ModelEntry]:
    found = []
    for key, label, run_names in CANDIDATES:
        path = weights / f"{key}.pt" if (weights / f"{key}.pt").exists() else None
        for r in run_names:
            if path is None:
                path = _ckpt_for_run(runs / r)
        if path is not None:
            found.append(ModelEntry(key, label, path))
    ens = results / "ensemble.json"
    if ens.exists():
        try:
            d = json.loads(ens.read_text(encoding="utf-8"))
            have = {m.key for m in found}
            if d.get("adopted") and d.get("members") and set(d["members"]) <= have:
                found.append(ModelEntry("ensemble", "Ensemble (ViT-L/14 + EfficientNet)", None, list(d["members"])))
        except (OSError, ValueError):
            pass
    return found


def load_calibration(key: str, results: Path = RESULTS) -> dict:
    """{'temperature', 'low', 'high', 'fitted'}; the fallback when Phase 4 has not written calibration.json."""
    path = results / "calibration.json"
    if path.exists():
        try:
            c = json.loads(path.read_text(encoding="utf-8")).get(key)
            if c:
                return dict(temperature=float(c["temperature"]), low=float(c["low"]), high=float(c["high"]),
                            fitted=True)
        except (OSError, ValueError, KeyError):
            pass
    return dict(temperature=1.0, low=FALLBACK_BAND[0], high=FALLBACK_BAND[1], fitted=False)


def measured_auc(results: Path = RESULTS, key: str | None = None) -> dict:
    """{'celebdf': x, 'wild': y} of model `key`, written by Phase 4 (results\\app_numbers.json), or {} until then."""
    path = results / "app_numbers.json"
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if key is not None and "models" in d:
            d = d["models"].get(key, {})
        return {k: float(d[k]) for k in ("celebdf", "wild") if k in d}
    except (OSError, ValueError, TypeError):
        return {}


def verdict(score: float, frame_probs: np.ndarray, low: float, high: float) -> dict:
    """Three bands and the confidence label (PLAN 11)."""
    if score >= high:
        label, agree = "fake", float(np.mean(frame_probs >= 0.5))
    elif score <= low:
        label, agree = "real", float(np.mean(frame_probs < 0.5))
    else:
        return dict(label="inconclusive", confidence=None, agree=None,
                    sentence="The model is unsure. Some frames look manipulated and others do not.")
    margin = score - high if label == "fake" else low - score
    confidence = "High" if margin >= 0.15 and agree >= 0.8 else "Medium"
    sentence = ("Signs of face manipulation were found in most analysed frames." if label == "fake"
                else "No significant signs of face manipulation were found.")
    return dict(label=label, confidence=confidence, agree=agree, sentence=sentence)


def to_tensor(crops_bgr: list[np.ndarray]) -> torch.Tensor:
    """BGR crops -> float RGB in [0, 1], after a JPEG 95 round trip so they match the training crops."""
    out = []
    for c in crops_bgr:
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        c = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else c
        out.append(torch.from_numpy(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    return torch.stack(out).float().div_(255)


def jpeg_bytes(img_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("could not encode a JPEG")
    return buf.tobytes()


class Analyzer:
    """Holds the face detector and loaded models. One analysis at a time (the caller serialises calls)."""

    def __init__(self, device: str | None = None, detector=None, loader=None, runs: Path = RUNS,
                 weights: Path = WEIGHTS, results: Path = RESULTS):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = detector
        self._loader = loader
        self.runs, self.weights, self.results = runs, weights, results
        self._models = {}

    # -- models
    def models(self) -> list[ModelEntry]:
        return find_models(self.runs, self.weights, self.results)

    def detector(self):
        if self._detector is None:
            from ..data.faces import FaceDetector
            self._detector = FaceDetector(device=self.device)
        return self._detector

    def model(self, entry: ModelEntry):
        if entry.key not in self._models:
            if self._loader is not None:
                self._models[entry.key] = self._loader(entry)
            else:
                from ..train.loop import load_model_checkpoint
                self._models[entry.key] = load_model_checkpoint(entry.path, self.device)[0]
            while len(self._models) > 2:                # 6 GB VRAM: keep at most two models resident
                self._models.pop(next(iter(self._models)))
        return self._models[entry.key]

    def _frame_probs(self, model, x: torch.Tensor, temperature: float, progress=None) -> np.ndarray:
        probs = []
        for i in range(0, len(x), 8):
            xb = x[i:i + 8].to(self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=self.device == "cuda"):
                logits, _ = model(xb)
            probs.append(F.softmax(logits.float() / temperature, dim=1)[:, 1].cpu())
            if progress:
                progress("analyze", min(1.0, (i + 8) / len(x)))
        return torch.cat(probs).numpy()

    # -- one upload
    def analyze(self, video_path: Path, model_key: str, progress=None) -> dict:
        say = progress or (lambda step, frac: None)
        t0 = time.time()
        entries = {m.key: m for m in self.models()}
        if model_key not in entries:
            raise AnalysisError("unreadable", f"model {model_key!r} is not available")
        entry = entries[model_key]

        say("extract", 0.0)
        frames, idx = read_frames_file(str(video_path), N_FRAMES)
        if len(frames) < 4:
            raise AnalysisError("unreadable", "This file can't be analyzed.")
        say("extract", 1.0)

        say("detect", 0.0)
        boxes = self.detector()(frames)
        if boxes is None:
            raise AnalysisError("no_face", "No face found.")
        crops = crops_for(frames, boxes, frames[0].shape[:2])
        say("detect", 1.0)

        x = to_tensor(crops)
        members = entry.members or [entry.key]
        per_member, cals = [], {}
        for k in members:
            m_entry = entries[k]
            cals[k] = load_calibration(k, self.results)
            per_member.append(self._frame_probs(self.model(m_entry), x, cals[k]["temperature"], say))
        probs = np.mean(per_member, axis=0)
        say("analyze", 1.0)

        say("result", 0.0)
        cal = load_calibration(entry.key, self.results) if entry.key == "ensemble" else cals[members[0]]
        score = float(np.mean(probs))
        v = verdict(score, probs, cal["low"], cal["high"])
        order = np.argsort(-np.abs(probs - 0.5))[:N_HEATMAPS]    # the most decisive frames
        order = sorted(order.tolist())
        from ..explain.gradcam import gradcam, overlay
        cam_model = self.model(entries[members[0]])
        cams = gradcam(cam_model, x[order].to(self.device))
        faces = [dict(frame=int(i) + 1, prob=float(probs[i]), image=jpeg_bytes(overlay(crops[i], cams[j], strength=float(probs[i]))))
                 for j, i in enumerate(order)]
        say("result", 1.0)
        return dict(
            model=entry.label, model_key=entry.key, score=score, frame_probs=[float(p) for p in probs],
            frame_index=[int(i) for i in idx], n_frames=len(probs), low=cal["low"], high=cal["high"],
            calibrated=cal["fitted"], faces=faces, seconds=round(time.time() - t0, 1),
            auc=measured_auc(self.results, entry.key), **v)
