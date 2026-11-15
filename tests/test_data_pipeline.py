import json
import logging
import zipfile

import cv2
import numpy as np
import pytest

from unmask.data import batch
from unmask.data.dfdc_zip import build_groups, evenly_spaced
from unmask.data.extract import ExtractConfig, extract_part
from unmask.data.faces import choose_track, fill_missing, square_crop
from unmask.data.gates import gates_pass, make_sample_sheet, run_gates
from unmask.data.index import read_index
from unmask.data.score import local_noise_score
from unmask.data.splits import grouped_split, groups_from_rows

LOG = logging.getLogger("test")


# ---------------------------------------------------------------- pairing
def test_build_groups_pairs_fakes_with_their_original():
    meta = {
        "r1.mp4": {"label": "REAL"},
        "r2.mp4": {"label": "REAL"},
        "r3.mp4": {"label": "REAL"},  # listed but not in zip
        "f1.mp4": {"label": "FAKE", "original": "r1.mp4"},
        "f2.mp4": {"label": "FAKE", "original": "r1.mp4"},
        "f3.mp4": {"label": "FAKE", "original": "r2.mp4"},
        "f4.mp4": {"label": "FAKE", "original": "rX.mp4"},  # original in another part
        "f5.mp4": {"label": "FAKE", "original": "r2.mp4"},  # not in zip
    }
    members = {"p/" + k for k in ["r1.mp4", "r2.mp4", "f1.mp4", "f2.mp4", "f3.mp4", "f4.mp4"]}
    reals, fakes_of, info = build_groups(meta, members, "p/")
    assert reals == ["r1.mp4", "r2.mp4"]
    assert fakes_of == {"r1.mp4": ["f1.mp4", "f2.mp4"], "r2.mp4": ["f3.mp4"]}
    assert info["reals_missing"] == ["r3.mp4"]
    assert info["fakes_orphan"] == 1 and info["fakes_missing"] == 1


def test_evenly_spaced():
    assert evenly_spaced(300, 16)[0] == 0 and evenly_spaced(300, 16)[-1] == 299
    assert len(evenly_spaced(300, 16)) == 16
    assert evenly_spaced(0, 16) == []


# ---------------------------------------------------------------- splits
def test_grouped_split_keeps_groups_whole_and_is_deterministic():
    rows = [dict(label="REAL", status="ok", video=f"r{i}.mp4", part=p) for p in (0, 1) for i in range(50)]
    groups = groups_from_rows(rows)
    tr, va = grouped_split(groups, 0.2, seed=0)
    assert len(va) == 20 and len(tr) == 80 and not set(tr) & set(va)
    assert (tr, va) == grouped_split(list(reversed(groups)), 0.2, seed=0)
    assert grouped_split(groups, 0.2, seed=1)[1] != va


# ---------------------------------------------------------------- score
def _crops(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n)]


def test_score_identical_crops_is_flagged():
    real = _crops()
    noisy = [np.clip(c.astype(int) + np.random.default_rng(1).integers(-2, 3, c.shape), 0, 255).astype(np.uint8)
             for c in real]
    score, noise = local_noise_score(noisy, real)
    assert score < 2.5 and noise > 0
    score0, _ = local_noise_score(real, real)
    assert score0 < 2.5


def test_score_local_face_change_is_not_flagged():
    real = _crops()
    rng = np.random.default_rng(2)
    fake = []
    for c in real:
        f = np.clip(c.astype(int) + rng.integers(-2, 3, c.shape), 0, 255).astype(np.uint8)
        f[90:140, 90:140] = 255 - f[90:140, 90:140]  # strong change in the face centre
        fake.append(f)
    score, _ = local_noise_score(fake, real)
    assert score > 10


# ---------------------------------------------------------------- faces
def test_choose_track_follows_first_person_and_fills_gaps():
    a = np.array([100, 100, 200, 200.0])
    b = np.array([500, 100, 600, 200.0])
    dets = [
        None,
        (np.stack([b, a]), np.array([0.90, 0.99])),       # a most confident
        (np.stack([b, a + 5]), np.array([0.999, 0.95])),  # b more confident, but a is nearer
        (np.stack([b]), np.array([0.99])),                # only b, too far: treated as a miss
        None,
    ]
    track = choose_track(dets)
    assert track[1] == a.tolist() and track[2] == (a + 5).tolist()
    assert track[0] == track[1] and track[3] == track[2] and track[4] == track[2]
    assert choose_track([None, None]) is None
    assert fill_missing([None, [1, 2, 3, 4]]) == [[1, 2, 3, 4], [1, 2, 3, 4]]


def test_square_crop_size_and_clipping():
    frame = np.zeros((360, 640, 3), np.uint8)
    assert square_crop(frame, [600, 300, 700, 400]).shape == (224, 224, 3)
    assert square_crop(frame, [0, 0, 640, 360]).shape == (224, 224, 3)


# ---------------------------------------------------------------- synthetic zip for extraction, gates, resume
def _write_video(path, n=20, invert_centre=False, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(n):
        f = np.roll(base, i, axis=1).copy()
        if invert_centre:
            f[45:75, 65:95] = 255 - f[45:75, 65:95]
        vw.write(f)
    vw.release()


@pytest.fixture
def synthetic_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    meta = {}
    for r in range(4):
        real = f"real{r}.mp4"
        _write_video(src / real, seed=r)
        meta[real] = {"label": "REAL", "split": "train"}
        for k in range(3):
            fake = f"fake{r}_{k}.mp4"
            if k == 0:  # identical copy -> near-identical, flagged
                (src / fake).write_bytes((src / real).read_bytes())
            else:
                _write_video(src / fake, invert_centre=True, seed=r)
            meta[fake] = {"label": "FAKE", "split": "train", "original": real}
    zp = tmp_path / "dfdc_train_part_07.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("dfdc_train_part_7/metadata.json", json.dumps(meta))
        for p in src.iterdir():
            zf.write(p, f"dfdc_train_part_7/{p.name}")
    return zp


def stub_detector(frames):
    return [[60.0, 40.0, 100.0, 80.0] for _ in frames]


def test_extract_gates_and_resume(synthetic_zip, tmp_path):
    out = tmp_path / "crops" / "part_07"
    cfg = ExtractConfig(frames=8, max_fakes=3, quality=95)  # 3 > 2 usable available: every fake is read
    s1 = extract_part(synthetic_zip, 7, out, cfg, stub_detector, log=LOG, limit_groups=2, show_bar=False)
    rows1 = read_index(out / "index.csv")
    assert {r["video"] for r in rows1 if r["label"] == "REAL"} == {"real0.mp4", "real1.mp4"}
    assert s1["real_ok"] == 2

    s2 = extract_part(synthetic_zip, 7, out, cfg, stub_detector, log=LOG, show_bar=False)
    rows = read_index(out / "index.csv")
    assert s2["real_ok"] == 2  # only the two remaining groups were processed
    reals = [r["video"] for r in rows if r["label"] == "REAL"]
    assert sorted(reals) == [f"real{i}.mp4" for i in range(4)] and len(reals) == len(set(reals))

    fakes = [r for r in rows if r["label"] == "FAKE"]
    flagged = [r for r in fakes if r["flagged"] == "1"]
    usable = [r for r in fakes if r["flagged"] == "0"]
    assert all(r["video"].endswith("_0.mp4") for r in flagged)
    assert len(usable) == 8 and len(flagged) == 4  # flagged ones are saved and marked, not dropped
    assert all(r["part"] == "7" for r in rows)

    make_sample_sheet(out, rows)
    results = run_gates(out, rows, n_decode=50)
    by = {g["gate"]: g for g in results}
    assert all(by[g]["ok"] for g in ["G1", "G2", "G4", "G5", "G6", "G7"]), results
    # 4 flagged of 12 scored = 33% -> G3 fails on this synthetic part
    assert not by["G3"]["ok"] and not gates_pass(results)


def test_gates_detect_missing_real_and_missing_jpeg(synthetic_zip, tmp_path):
    out = tmp_path / "crops" / "part_07"
    cfg = ExtractConfig(frames=8, max_fakes=2)
    extract_part(synthetic_zip, 7, out, cfg, stub_detector, log=LOG, limit_groups=3, show_bar=False)
    rows = read_index(out / "index.csv")
    next((out / "real0").glob("*.jpg")).unlink()
    by = {g["gate"]: g for g in run_gates(out, rows, n_decode=10)}
    assert not by["G1"]["ok"]  # real3 has no row
    assert not by["G4"]["ok"]  # real0 is missing a frame
    assert not by["G6"]["ok"]  # no sample sheet yet


# ---------------------------------------------------------------- batch bookkeeping
def test_next_message_follows_batch_table():
    assert "part_00" in batch.next_message(set())
    assert "part_48" in batch.next_message({0, 1})
    assert "part_02" in batch.next_message({0, 1, 48, 49})
    assert "all planned" in batch.next_message({p for b in batch.BATCHES for p in b})


def test_delete_done_zips_needs_pipeline_ok_and_done(tmp_path):
    zdir, cdir, ok = tmp_path / "zip data", tmp_path / "crops", tmp_path / "pipeline_ok.json"
    zdir.mkdir()
    for p in (0, 1):
        (zdir / f"dfdc_train_part_{p:02d}.zip").write_bytes(b"x" * 10)
    (zdir / "other.zip").write_bytes(b"y")
    (cdir / "part_00").mkdir(parents=True)
    (cdir / "part_00" / "DONE.json").write_text(json.dumps({"gates_pass": True}))

    deleted, status = batch.delete_done_zips(LOG, zdir, cdir, ok)
    assert deleted == [] and "kept" in status and (zdir / "dfdc_train_part_00.zip").exists()

    ok.write_text("{}")
    deleted, _ = batch.delete_done_zips(LOG, zdir, cdir, ok)
    assert len(deleted) == 1 and not (zdir / "dfdc_train_part_00.zip").exists()
    assert (zdir / "dfdc_train_part_01.zip").exists() and (zdir / "other.zip").exists()
