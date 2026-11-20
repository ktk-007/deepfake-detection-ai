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


def test_several_holdout_groups_excluded_with_all_fakes(tmp_path, crops):
    h = tmp_path / "hold.json"
    h.write_text(json.dumps({"groups": [
        {"part": 0, "real": "p0r2.mp4", "fakes_of_real": ["p0r2f0.mp4", "p0r2f1.mp4"]},
        {"part": 1, "real": "p1r5.mp4", "demo": ["p1r5.mp4", "p1r5f0.mp4", "p1r5f1.mp4"]},
        {"part": 18, "real": "notextracted.mp4"},           # part not trained on here: ignored, no error
    ]}))
    args = _args(tmp_path, crops, "--name", "h2", "--epochs", "1")
    args[args.index("--holdout") + 1] = str(h)
    train_script.main(args)
    cfg = json.loads((tmp_path / "runs" / "h2" / "config.json").read_text())
    assert cfg["holdout_excluded"] == ["00/p0r2.mp4", "01/p1r5.mp4"]
    assert cfg["train_reals"] == 14 and cfg["train_fakes"] == 28   # 16 reals, 32 fakes minus 2 whole groups


def test_holdout_group_missing_from_trained_part_is_an_error(tmp_path, crops):
    h = tmp_path / "hold.json"
    h.write_text(json.dumps({"groups": [{"part": 1, "real": "typo.mp4"}]}))
    args = _args(tmp_path, crops, "--name", "h3", "--epochs", "1")
    args[args.index("--holdout") + 1] = str(h)
    with pytest.raises(SystemExit, match="hold-out group"):
        train_script.main(args)


def test_parts_dev_and_every_config_loads():
    cfg = train_script.resolve_config(train_script.parse_args(["--parts", "dev"]))
    assert cfg["parts"] == [0, 1, 2, 3, 4, 5]
    for y in sorted(y for y in (ROOT / "configs").glob("*.yaml") if not y.name.startswith("queue_")):
        cfg = train_script.resolve_config(train_script.parse_args(["--config", str(y)]))
        assert cfg["epochs"] > 0, y.name
        if cfg["parts"] is not None:
            assert not {8, 9} & set(cfg["parts"]), y.name


# ---------------------------------------------------------------- Gate 2b: robust preset, views, eval, averaging
def test_robust_preset_steps_keep_shape_and_dtype():
    from unmask.data.augment import CropZoom, DownUp, GaussNoise, build_augment
    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    torch.manual_seed(0)
    aug = build_augment("robust")
    for _ in range(20):
        out = aug(img)
        assert out.shape == img.shape and out.dtype == torch.uint8
    for step in (CropZoom(p=1.0), DownUp(p=1.0), GaussNoise(p=1.0)):
        out = step(img)
        assert out.shape == img.shape and out.dtype == torch.uint8 and not torch.equal(out, img)
    assert torch.equal(CropZoom(p=0.0)(img), img) and torch.equal(DownUp(p=0.0)(img), img)
    flat = torch.full((3, 224, 224), 100, dtype=torch.uint8)
    assert torch.equal(CropZoom(p=1.0, zoom=(1.3, 1.3), shift=0.0)(flat), flat)  # zoom of a flat image is flat


def test_robust_half_halves_only_the_four_new_steps():
    from unmask.data.augment import CropZoom, DownUp, GaussNoise, GridDropout, build_augment

    def probs(preset):
        steps = build_augment(preset).transforms
        p = {type(s).__name__: s.p for s in steps if isinstance(s, (CropZoom, DownUp, GaussNoise, GridDropout))}
        p["second_jpeg"] = [s.p for s in steps if type(s).__name__ == "RandomApply"
                            and type(s.transforms[0]).__name__ == "JPEG"][0]
        p["n_steps"] = len(steps)
        return p

    full, half = probs("robust"), probs("robust_half")
    assert full == {"CropZoom": 0.5, "DownUp": 0.3, "GaussNoise": 0.2, "GridDropout": 0.3,
                    "second_jpeg": 0.2, "n_steps": 10}
    assert half == {"CropZoom": 0.25, "DownUp": 0.15, "GaussNoise": 0.1, "GridDropout": 0.3,
                    "second_jpeg": 0.1, "n_steps": 10}
    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    assert build_augment("robust_half")(img).shape == img.shape
    cfg = train_script.resolve_config(train_script.parse_args(["--aug", "robust_half"]))
    assert cfg["aug"] == "robust_half"


def test_views_are_deterministic_and_change_the_image():
    from unmask.eval.views import VIEWS, degraded, tight, view_fn
    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    assert view_fn("clean") is None and set(VIEWS) == {"clean", "tight", "degraded"}
    for f in (tight, degraded):
        a, b = f(img), f(img)
        assert a.shape == img.shape and a.dtype == torch.uint8 and torch.equal(a, b) and not torch.equal(a, img)
    grad = torch.arange(224, dtype=torch.uint8).repeat(3, 224, 1)     # horizontal ramp 0..223
    t = tight(grad)
    assert t[0, 0, 0] > grad[0, 0, 0] and t[0, 0, -1] < grad[0, 0, -1]  # the zoom drops the outer columns


def test_append_csv_widens_header_and_keeps_old_rows(tmp_path):
    from unmask.train.loop import RESULT_FIELDS, append_csv, update_result
    p = tmp_path / "results.csv"
    old = RESULT_FIELDS[:RESULT_FIELDS.index("note") + 1]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({k: "x" for k in old} | {"name": "gate2_b16", "val_auc": "0.75406"})
    append_csv(p, RESULT_FIELDS, {"name": "new_run", "val_auc": 0.8})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [r["name"] for r in rows] == ["gate2_b16", "new_run"]
    assert rows[0]["val_auc"] == "0.75406" and rows[0]["val_robust"] == ""
    assert update_result(p, "gate2_b16", {"val_robust": 0.7}) and not update_result(p, "nope", {"val_robust": 1})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows[0]["val_robust"] == "0.70000" and rows[0]["arch"] == "x" and rows[1]["val_robust"] == ""


def test_low_disk_stops_after_the_epoch(tmp_path, crops, monkeypatch):
    monkeypatch.setattr(train_script, "disk_free_gb", lambda: 5.0)
    train_script.main(_args(tmp_path, crops, "--name", "d", "--epochs", "3", "--patience", "0"))
    ck = torch.load(tmp_path / "runs" / "d" / "last.pt", weights_only=False)
    assert ck["epoch"] == 0 and not (tmp_path / "results.csv").exists()  # stopped after epoch 1, no results row
    monkeypatch.setattr(train_script, "disk_free_gb", lambda: 50.0)
    train_script.main(_args(tmp_path, crops, "--name", "d", "--resume"))
    assert torch.load(tmp_path / "runs" / "d" / "last.pt", weights_only=False)["epoch"] == 2


def test_eval_val_and_average_ckpt(tmp_path, crops):
    import importlib.util as iu
    sys_path = str(ROOT / "scripts")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    ev = iu.module_from_spec(iu.spec_from_file_location("eval_val", ROOT / "scripts" / "eval_val.py"))
    ev.__spec__.loader.exec_module(ev)
    av = iu.module_from_spec(iu.spec_from_file_location("average_ckpt", ROOT / "scripts" / "average_ckpt.py"))
    av.__spec__.loader.exec_module(av)

    train_script.main(_args(tmp_path, crops, "--name", "r", "--epochs", "4", "--patience", "0", "--lr", "3e-3"))
    run = tmp_path / "runs" / "r"
    assert sorted(p.name for p in run.glob("ep_*.pt")) == ["ep_01.pt", "ep_02.pt", "ep_03.pt", "ep_04.pt"]
    common = ["--runs-dir", str(tmp_path / "runs"), "--crops-dir", str(crops), "--results",
              str(tmp_path / "results.csv"), "--device", "cpu"]
    ev.main(["--run", "r", "--frames", "4", *common])
    res = json.loads((run / "eval_val_best.json").read_text())
    assert set(res) >= {"clean", "tight", "degraded", "val_robust"} and res["n_frames"] == 6 * 3 * 4
    assert res["val_robust"] == pytest.approx(sum(res[v]["auc"] for v in ("clean", "tight", "degraded")) / 3)
    row = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))[-1]
    assert row["name"] == "r" and float(row["val_robust"]) == pytest.approx(res["val_robust"], abs=1e-4)

    dec = av.main(["--run", "r", "--k", "3", *common])
    assert len(dec["epochs"]) == 3 and (run / "avg.pt").exists() and (run / "eval_val_avg.json").exists()
    assert dec["use_avg"] == (dec["avg_val_robust"] - dec["best_val_robust"] >= 0.003)
    avg = torch.load(run / "avg.pt", weights_only=False)
    eps = [torch.load(run / f"ep_{e:02d}.pt", weights_only=False)["trainable"] for e in dec["epochs"]]
    k = next(iter(avg["trainable"]))
    assert torch.allclose(avg["trainable"][k], torch.stack([e[k].float() for e in eps]).mean(0))
    row = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))[-1]
    assert row["avg_used"] in ("True", "False") and row["avg_epochs"] == " ".join(map(str, dec["epochs"]))


def test_show_aug_writes_a_grid_per_preset(tmp_path, crops):
    import importlib.util as iu
    sa = iu.module_from_spec(iu.spec_from_file_location("show_aug", ROOT / "scripts" / "show_aug.py"))
    sa.__spec__.loader.exec_module(sa)
    paths = sa.main(["--crops-dir", str(crops), "--out-dir", str(tmp_path)])
    assert [p.name for p in paths] == ["aug_none.jpg", "aug_light.jpg", "aug_strong.jpg", "aug_robust.jpg",
                                       "aug_robust_half.jpg"]
    img = cv2.imread(str(paths[-1]))
    assert img.shape[:2] == (4 * 32, 4 * 32)   # 4 x 4 tiles of the 32 px synthetic crops


# ---------------------------------------------------------------- freeze and the night queue (PLAN 10.3)
def test_freeze_in_the_past_stops_after_the_first_epoch_and_keeps_best(tmp_path, crops):
    train_script.main(_args(tmp_path, crops, "--name", "fz", "--epochs", "3", "--freeze", "2000-01-01 00:00"))
    run = tmp_path / "runs" / "fz"
    assert (run / "best.pt").exists()
    rows = list(csv.DictReader(open(run / "metrics.csv", encoding="utf-8")))
    assert len(rows) == 1                                      # stopped after epoch 1 of 3
    res = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert res[-1]["name"] == "fz" and res[-1]["epochs_run"] == "1"   # counts as a final model (best.pt)


def _queue_module():
    spec = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "run_queue.py")
    rq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rq)
    return rq


def test_run_queue_order_freeze_resume_and_planned_not_run(tmp_path):
    from datetime import datetime, timedelta
    rq = _queue_module()
    q = tmp_path / "q.yaml"
    q.write_text(
        "freeze: \"2026-11-17 05:00\"\n"
        "runs:\n"
        "  - {name: big, config: configs/e7_l14.yaml, set: {epochs: 4, lr: 2.0e-4, grad_ckpt: true}}\n"
        "  - {name: cnn, config: configs/e3_effb4.yaml, set: {epochs: 6}}\n"
        "  - {name: late, config: configs/e1_linear_probe.yaml}\n", encoding="utf-8")
    runs = tmp_path / "runs"
    (runs / "cnn").mkdir(parents=True)
    (runs / "cnn" / "last.pt").write_bytes(b"x")              # interrupted earlier -> resumed
    clock = [datetime(2026, 11, 17, 1, 0)]                    # each training run moves the clock 2 h
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        if cmd[1].endswith("train.py"):
            name = cmd[cmd.index("--name") + 1]
            (runs / name).mkdir(parents=True, exist_ok=True)
            (runs / name / "best.pt").write_bytes(b"x")
            clock[0] += timedelta(hours=2, minutes=1)        # 01:00 -> 03:01 -> 05:02 (past the freeze)
        return 0

    report = dict(rq.main([str(q), "--runs-dir", str(runs), "--results", str(tmp_path / "res.csv")],
                          now=lambda: clock[0], disk=lambda: 30.0, run_cmd=fake_run))
    scripts = [(Path(c[1]).name, c[c.index("--name") + 1] if "--name" in c else c[c.index("--run") + 1])
               for c in calls]
    assert scripts == [("train.py", "big"), ("eval_val.py", "big"), ("average_ckpt.py", "big"),
                       ("train.py", "cnn"), ("eval_val.py", "cnn")]          # no averaging for EfficientNet
    big = calls[0]
    assert big[big.index("--epochs") + 1] == "4" and big[big.index("--lr") + 1] == "0.0002"
    assert "--grad-ckpt" in big and big[big.index("--freeze") + 1] == "2026-11-17 05:00"
    assert "--resume" in calls[3] and "--config" not in calls[3]
    assert report["late"].startswith("planned, not run")


def test_run_queue_does_not_start_below_the_disk_floor(tmp_path):
    rq = _queue_module()
    q = tmp_path / "q.yaml"
    q.write_text("freeze: \"2099-01-01 05:00\"\nruns:\n  - {name: a, config: configs/e2_b16_ln.yaml}\n",
                 encoding="utf-8")
    calls = []
    report = dict(rq.main([str(q), "--runs-dir", str(tmp_path / "runs"), "--results", str(tmp_path / "r.csv")],
                          disk=lambda: 12.0, run_cmd=lambda c: calls.append(c) or 0, sleep=lambda s: None))
    assert not calls and "free disk 12.0 GB" in report["a"]


def test_queue_finals_file_loads():
    rq = _queue_module()
    freeze, runs = rq.load_queue(ROOT / "configs" / "queue_finals.yaml")
    assert [r["name"] for r in runs[:4]] == ["e2_b16", "e7_l14", "e1_linear_probe", "e3_effb4"]
    assert freeze.hour == 5
    for r in runs:
        cfg = train_script.resolve_config(train_script.parse_args(
            ["--config", str(ROOT / r["config"])] + rq.flags(r.get("set"))))
        assert cfg["epochs"] > 0 and cfg["loss"] == "ce", r["name"]


def test_run_queue_continues_after_a_failed_or_crashing_job(tmp_path):
    rq = _queue_module()
    q = tmp_path / "q.yaml"
    q.write_text('freeze: "2099-01-01 05:00"\nruns:\n'
                 "  - {name: bad, config: configs/e2_b16_ln.yaml}\n"
                 "  - {name: boom, config: configs/e2_b16_ln.yaml}\n"
                 "  - {name: good, config: configs/e2_b16_ln.yaml}\n", encoding="utf-8")
    runs, calls = tmp_path / "runs", []

    def fake_run(cmd):
        calls.append(cmd)
        name = cmd[cmd.index("--name") + 1] if "--name" in cmd else cmd[cmd.index("--run") + 1]
        if name == "bad":
            return 1                                       # training exits with an error
        if name == "boom":
            raise OSError("launcher crashed")
        if cmd[1].endswith("train.py"):
            (runs / name).mkdir(parents=True, exist_ok=True)
            (runs / name / "best.pt").write_bytes(b"x")
        return 2 if cmd[1].endswith("eval_val.py") else 0  # a failed check is logged, not fatal

    report = dict(rq.main([str(q), "--runs-dir", str(runs), "--results", str(tmp_path / "r.csv")],
                          disk=lambda: 30.0, run_cmd=fake_run))
    assert report["bad"].startswith("FAILED") and "exit 1" in report["bad"]
    assert report["boom"].startswith("FAILED") and "OSError" in report["boom"]
    assert "check FAILED: eval_val exit 2" in report["good"]
    assert [Path(c[1]).name for c in calls][-3:] == ["train.py", "eval_val.py", "average_ckpt.py"]
