import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from unmask.data.index import INDEX_FIELDS
from unmask.train.loop import EarlyStop, eval_items, warmup_cosine

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)


def _make_part(crops: Path, part: int, n_real: int, frames: int, seed: int):
    """Synthetic part: reals are dark, fakes bright, so a tiny model can learn the task."""
    rng = np.random.default_rng(seed)
    d = crops / f"part_{part:02d}"
    rows = []
    for r in range(n_real):
        real = f"p{part}r{r}.mp4"
        vids = [(real, "REAL", "")] + [(f"p{part}r{r}f{k}.mp4", "FAKE", real) for k in range(2)]
        for v, lab, orig in vids:
            vd = d / Path(v).stem
            vd.mkdir(parents=True)
            base = 60 if lab == "REAL" else 180
            for i in range(frames):
                img = np.clip(rng.normal(base, 30, (32, 32, 3)), 0, 255).astype(np.uint8)
                cv2.imwrite(str(vd / f"f{i:02d}.jpg"), img)
            rows.append(dict(video=v, label=lab, original=orig, n_frames=frames, status="ok",
                             score="" if lab == "REAL" else "9.0", flagged="" if lab == "REAL" else "0",
                             noise="", part=part))
    with open(d / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (d / "DONE.json").write_text(json.dumps({"gates_pass": True}))


@pytest.fixture()
def crops(tmp_path):
    c = tmp_path / "crops"
    _make_part(c, 0, 8, 16, 0)
    _make_part(c, 1, 8, 16, 1)
    _make_part(c, 8, 6, 32, 8)
    _make_part(c, 9, 4, 32, 9)
    return c


def _args(tmp_path, crops, *extra):
    return ["--arch", "tiny", "--device", "cpu", "--workers", "0", "--crops-dir", str(crops),
            "--runs-dir", str(tmp_path / "runs"), "--results", str(tmp_path / "results.csv"),
            "--holdout", str(tmp_path / "no_holdout.json"),
            "--batch", "8", "--frames-per-video", "2", "--val-frames", "4", "--val-batch", "16", *extra]


def test_warmup_cosine_shape():
    f = warmup_cosine(total_steps=100, warmup_steps=10, base_lr=1e-3, min_lr=1e-6)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(1e-3)
    assert all(f(i) >= f(i + 1) for i in range(10, 100))


def test_early_stop_smoothed():
    s = EarlyStop(patience=2, smooth=3)
    assert s.update(0, 0.6) == (True, False)
    assert s.update(1, 0.7) == (True, False)
    assert s.update(2, 0.5) == (False, False)      # mean(.6,.7,.5)=.60 < best smoothed .65: 1 bad epoch
    assert s.update(3, 0.5) == (False, True)       # mean(.7,.5,.5)=.57: 2 bad epochs -> stop
    assert s.best_epoch == 1 and s.best_raw == 0.7
    s2 = EarlyStop(patience=2, smooth=3)
    for e, a in enumerate([0.6, 0.65, 0.5, 0.7]):  # a noisy dip followed by recovery does not stop
        _, stop = s2.update(e, a)
    assert not stop
    assert EarlyStop(patience=0).update(0, math.nan) == (False, False)


def test_config_merge_order(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("arch: effb4\nepochs: 7\naug: none\n")
    a = train_script.parse_args(["--config", str(y), "--epochs", "3", "--parts", "1,0"])
    cfg = train_script.resolve_config(a)
    assert cfg["arch"] == "effb4" and cfg["lr"] == 1e-4 and cfg["loss"] == "ce"   # arch defaults
    assert cfg["aug"] == "none" and cfg["epochs"] == 3 and cfg["parts"] == [0, 1]  # yaml, then CLI
    y.write_text("bogus: 1\n")
    with pytest.raises(SystemExit):
        train_script.resolve_config(train_script.parse_args(["--config", str(y)]))


def test_eval_items_frame_subset(crops):
    from unmask.data.dataset import load_groups
    g = list(load_groups([8], crops).values())
    assert len(eval_items(g, 4)) == 6 * 3 * 4 and len(eval_items(g, 0)) == 6 * 3 * 32


@pytest.mark.parametrize("parts", ["0,8", "1,9"])
def test_refuses_validation_or_test_part_in_training(tmp_path, crops, parts):
    with pytest.raises(ValueError):
        train_script.main(_args(tmp_path, crops, "--name", "bad", "--parts", parts))


def test_train_stop_resume_and_outputs(tmp_path, crops):
    base = _args(tmp_path, crops, "--name", "t", "--epochs", "3", "--patience", "0", "--lr", "3e-3")
    train_script.main(base + ["--stop-after", "1"])
    run = tmp_path / "runs" / "t"
    ck1 = torch.load(run / "last.pt", weights_only=False)
    assert ck1["epoch"] == 0 and len(ck1["history"]) == 1
    assert not (tmp_path / "results.csv").exists()          # incomplete run: no results row
    with pytest.raises(SystemExit):                          # refuses to overwrite an existing run
        train_script.main(base)

    train_script.main(_args(tmp_path, crops, "--name", "t", "--resume"))
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["epoch"] == 2 and [h["epoch"] for h in ck["history"]] == [1, 2, 3]
    assert ck["scheduler"]["last_epoch"] == 3 * ck1["scheduler"]["last_epoch"]  # schedule continued, not restarted
    rows = list(csv.DictReader(open(run / "metrics.csv", encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    best = torch.load(run / "best.pt", weights_only=False)
    assert best["kind"] == "best" and "optimizer" not in best
    res = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert len(res) == 1 and res[0]["train_parts"] == "0 1" and res[0]["epochs_run"] == "3"
    assert float(res[0]["val_auc"]) > 0.9                   # the synthetic task is easy
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["parts"] == [0, 1] and 9 not in cfg["parts"]
    scores = list(csv.DictReader(open(run / "val_best.csv", encoding="utf-8")))
    assert len(scores) == 6 * 3 and all(s["video"].startswith("08/") for s in scores)


def test_app_holdout_group_excluded_from_training(tmp_path, crops):
    h = tmp_path / "hold.json"
    h.write_text(json.dumps({"part": 1, "real": "p1r3.mp4", "fake": "p1r3f0.mp4"}))
    args = _args(tmp_path, crops, "--name", "h", "--epochs", "1")
    args[args.index("--holdout") + 1] = str(h)
    train_script.main(args)
    cfg = json.loads((tmp_path / "runs" / "h" / "config.json").read_text())
    assert cfg["holdout_excluded"] == ["01/p1r3.mp4"] and cfg["train_reals"] == 15


def test_parts_dev_and_every_config_loads():
    cfg = train_script.resolve_config(train_script.parse_args(["--parts", "dev"]))
    assert cfg["parts"] == [0, 1, 2, 3, 4, 5]
    for y in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = train_script.resolve_config(train_script.parse_args(["--config", str(y)]))
        assert cfg["epochs"] > 0, y.name
        if cfg["parts"] is not None:
            assert not {8, 9} & set(cfg["parts"]), y.name
