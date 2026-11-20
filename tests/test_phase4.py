"""Phase 4: calibration, band, bootstrap, the part-08 lock and the once-only test scoring (tiny model, CPU)."""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from unmask.data.index import INDEX_FIELDS
from unmask.eval.phase4 import Scores, bootstrap_auc, choose_band, ece, fit_temperature, load_set

ROOT = Path(__file__).resolve().parent.parent


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _img(path, base, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(rng.normal(base, 30, (32, 32, 3)), 0, 255).astype(np.uint8))


def _dfdc(crops, part, n_real, frames, rng):
    d = crops / f"part_{part:02d}"
    rows = []
    for r in range(n_real):
        real = f"p{part}r{r}.mp4"
        for v, lab, orig in [(real, "REAL", "")] + [(f"p{part}r{r}f{k}.mp4", "FAKE", real) for k in range(2)]:
            for i in range(frames):
                _img(d / Path(v).stem / f"f{i:02d}.jpg", 60 if lab == "REAL" else 180, rng)
            rows.append(dict(video=v, label=lab, original=orig, n_frames=frames, status="ok",
                             score="" if lab == "REAL" else "9", flagged="" if lab == "REAL" else "0", part=part))
    with open(d / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in INDEX_FIELDS} for r in rows])
    (d / "DONE.json").write_text("{}")


@pytest.fixture
def crops(tmp_path):
    rng = np.random.default_rng(0)
    c = tmp_path / "crops"
    for part, n in ((0, 8), (1, 8), (8, 8), (9, 6)):
        _dfdc(c, part, n, 8, rng)
    cd = c / "celebdf_v2_test"
    rows = []
    for i in range(10):
        lab = "FAKE" if i % 2 else "REAL"
        v = f"{'Celeb-synthesis' if lab == 'FAKE' else 'Celeb-real'}/id{i}_0000.mp4"
        for k in range(8):
            _img(cd / Path(v).stem / f"f{k:02d}.jpg", 180 if lab == "FAKE" else 60, rng)
        rows.append(dict(video=v, label=lab, n_frames=8, status="ok", part="celebdf"))
    with open(cd / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in INDEX_FIELDS} for r in rows])
    wd = c / "wild_test"
    wd.mkdir()
    with open(wd / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "label", "n_frames", "source_tar", "n_source_frames", "status"])
        for i in range(12):
            lab = "FAKE" if i % 2 else "REAL"
            vid = f"{lab.lower()}_{i // 4}_{i}"
            for k in range(8):
                _img(wd / vid / f"f{k:02d}.jpg", 180 if lab == "FAKE" else 60, rng)
            w.writerow([vid, lab, 8, f"{lab.lower()}_test/{i // 4}.tar.gz", 50, "ok"])
    return c


def test_temperature_band_ece_bootstrap():
    rng = np.random.default_rng(1)
    y = np.array([0] * 50 + [1] * 50)
    wrong = rng.random(100) < 0.2                                          # 20% confidently wrong answers
    z = np.where((y == 1) ^ wrong, 6.0, -6.0) + rng.normal(0, 1, 100)    # -> over-confident logits
    s = Scores([str(i) for i in range(100)], y, [str(i) for i in range(100)], np.arange(101), z)
    t = fit_temperature(s)
    assert t > 1.0                                                        # softens over-confidence
    p = s.video_probs(t)
    assert choose_band(np.full(20, 0.5), np.array([0, 1] * 10))["fitted"] is False   # too few per class
    # overlapping classes: a middle band, with per-class error control
    q = np.concatenate([rng.uniform(0.0, 0.6, 200), rng.uniform(0.4, 1.0, 600)])     # 75% fake, like part 08
    yq = np.concatenate([np.zeros(200, int), np.ones(600, int)])
    band = choose_band(q, yq)
    assert band["fitted"] and band["low"] < band["high"]
    assert np.mean(q[yq == 0] > band["high"]) <= 0.05 and np.mean(q[yq == 1] < band["low"]) <= 0.05
    # the band does not move when the share of fakes changes (prevalence-independent)
    b2 = choose_band(np.concatenate([q[:200], q[200:350]]), np.concatenate([yq[:200], yq[200:350]]))
    assert abs(b2["high"] - band["high"]) < 1e-9
    # separated classes: the gap between them stays an Inconclusive zone
    sep = choose_band(np.concatenate([rng.uniform(0.0, 0.2, 40), rng.uniform(0.8, 1.0, 40)]), np.array([0] * 40 + [1] * 40))
    assert 0.15 < sep["low"] <= 0.2 and 0.8 <= sep["high"] < 0.85
    assert choose_band(np.full(20, 0.5), np.array([0, 1] * 10)) == dict(low=0.35, high=0.65, fitted=False)
    assert 0 <= ece(p, y) <= 1
    lo, hi = bootstrap_auc(y, p, [str(i // 2) for i in range(100)], n=200)
    assert lo <= hi <= 1.0


def test_load_sets_groups(crops):
    v08 = load_set("val08", crops)
    assert len(v08) == 24 and {v.group for v in v08 if v.label == 1} <= {v.group for v in v08 if v.label == 0}
    wild = load_set("wild", crops)
    assert len(wild) == 12 and len({v.group for v in wild}) == 6        # bootstrap by source tar
    cel = load_set("celebdf", crops)
    assert len(cel) == 10 and all(len(v.frames) == 8 for v in cel)


def test_phase4_val_locks_then_test_scores_each_set_once(tmp_path, crops):
    train = _mod("train")
    common = ["--arch", "tiny", "--device", "cpu", "--workers", "0", "--crops-dir", str(crops),
              "--runs-dir", str(tmp_path / "runs"), "--results", str(tmp_path / "results.csv"),
              "--holdout", str(tmp_path / "none.json"), "--batch", "8", "--frames-per-video", "2",
              "--val-frames", "4", "--val-batch", "16", "--epochs", "2", "--parts", "0,1"]
    train.main(common + ["--name", "e2_b16"])
    train.main(common + ["--name", "e3_effb4", "--seed", "1"])
    res = tmp_path / "res"
    pv, pt = _mod("phase4_val"), _mod("phase4_test")
    args = ["--device", "cpu", "--runs-dir", str(tmp_path / "runs"), "--crops-dir", str(crops), "--results", str(res)]
    with pytest.raises(SystemExit, match="phase4_val"):
        pt.main(["--device", "cpu", "--crops-dir", str(crops), "--results", str(res)])   # no lock -> refuses
    lock = pv.main(args)
    assert set(lock["models"]) == {"e2_b16", "e3_effb4"}
    assert (res / "phase4" / "choices.json").exists()
    cal = json.loads((res / "calibration.json").read_text())
    assert set(cal) >= {"vit_b16", "effb4"} and all("temperature" in v for v in cal.values())
    assert not any(p.name.startswith("test_") for p in (res / "phase4").iterdir())   # no test set opened yet

    m = pt.main(["--device", "cpu", "--crops-dir", str(crops), "--results", str(res), "--boot", "50"])
    assert set(m) >= {"e2_b16", "e3_effb4"} and set(m["e2_b16"]) == {"test09", "celebdf", "wild"}
    ledger = json.loads((res / "phase4" / "test_ledger.json").read_text())
    n_opened = len(ledger)
    assert (res / "final_table.md").exists() and (res / "app_numbers.json").exists()
    assert (res / "plots" / "roc_celebdf.svg").exists()
    pt.main(["--device", "cpu", "--crops-dir", str(crops), "--results", str(res), "--boot", "50"])
    assert len(json.loads((res / "phase4" / "test_ledger.json").read_text())) == n_opened   # never re-scored
