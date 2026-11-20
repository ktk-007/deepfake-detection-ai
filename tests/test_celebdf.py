import json
import zipfile

import cv2
import numpy as np
import pytest

from unmask.data.celebdf import (CelebConfig, extract_celebdf, folder_label, list_flag_meaning, parse_list,
                                 run_celeb_gates)
from unmask.data.index import read_index


def _write_video(path, n=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(n):
        vw.write(np.roll(base, i, axis=1).copy())
    vw.release()


def stub_detector(frames):
    return [[60.0, 40.0, 100.0, 80.0] for _ in frames]


@pytest.fixture
def celeb_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    listed = ["1 YouTube-real/00170.mp4", "1 Celeb-real/id0_0001.mp4", "0 Celeb-synthesis/id0_id1_0001.mp4",
              "0 Celeb-synthesis/id2_id3_0004.mp4"]
    zp = tmp_path / "Celeb-DF-v2.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("Celeb-DF-v2/List_of_testing_videos.txt", "\n".join(listed) + "\n")
        for k, line in enumerate(listed + ["0 Celeb-synthesis/id9_id8_0000.mp4"]):   # last one: not in the list
            rel = line.split()[1]
            p = src / f"v{k}.mp4"
            _write_video(p, seed=k)
            zf.write(p, f"Celeb-DF-v2/{rel}")
    return zp


def test_list_parsing_and_labels():
    e = parse_list("1 YouTube-real/a.mp4\n\n0 Celeb-synthesis/b.mp4\n1 Celeb-real/c.mp4\n1 YouTube-real/a.mp4\n")
    assert e == [(1, "YouTube-real/a.mp4"), (0, "Celeb-synthesis/b.mp4"), (1, "Celeb-real/c.mp4")]
    assert [folder_label(p) for _, p in e] == ["REAL", "FAKE", "REAL"]
    assert list_flag_meaning(e)["consistent"]
    assert not list_flag_meaning([(1, "Celeb-real/x.mp4"), (1, "Celeb-synthesis/y.mp4")])["consistent"]
    with pytest.raises(ValueError):
        parse_list("YouTube-real/a.mp4\n")


def test_extract_only_listed_videos_gates_and_resume(celeb_zip, tmp_path):
    out = tmp_path / "celebdf_v2_test"
    cfg = CelebConfig(frames=8)
    s1 = extract_celebdf(celeb_zip, out, cfg, stub_detector, limit=2, show_bar=False)
    assert s1["listed"] == 4 and len(read_index(out / "index.csv")) == 2
    s2 = extract_celebdf(celeb_zip, out, cfg, stub_detector, show_bar=False)
    rows = read_index(out / "index.csv")
    assert [r["video"] for r in rows] == ["YouTube-real/00170.mp4", "Celeb-real/id0_0001.mp4",
                                          "Celeb-synthesis/id0_id1_0001.mp4", "Celeb-synthesis/id2_id3_0004.mp4"]
    assert [r["label"] for r in rows] == ["REAL", "REAL", "FAKE", "FAKE"]
    assert all(r["status"] == "ok" and r["n_frames"] == "8" and r["part"] == "celebdf" for r in rows)
    assert not (out / "id9_id8_0000").exists()                       # not in the test list
    assert len(list((out / "00170").glob("*.jpg"))) == 8
    gates = run_celeb_gates(out, rows, 4, s2["flags"])
    assert all(ok for _, ok, _ in gates), gates


def test_no_face_is_recorded_and_c2_fails_when_most_are_lost(celeb_zip, tmp_path):
    out = tmp_path / "c"
    s = extract_celebdf(celeb_zip, out, CelebConfig(frames=8), lambda f: None, show_bar=False)
    rows = read_index(out / "index.csv")
    assert all(r["status"] == "no_face" for r in rows)
    gates = dict((n, ok) for n, ok, _ in run_celeb_gates(out, rows, 4, s["flags"]))
    assert gates["C1"] and not gates["C2"]


def test_prepare_script_writes_done_json(celeb_zip, tmp_path):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("prepare_celebdf", root / "scripts" / "prepare_celebdf.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "o"
    gates = m.main(["--zip", str(celeb_zip), "--out", str(out)], detector=stub_detector)
    assert all(ok for _, ok, _ in gates)
    done = json.loads((out / "DONE.json").read_text())
    assert done["listed"] == 4 and (out / "_samples.jpg").exists()
