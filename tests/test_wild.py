import io
import tarfile

import cv2
import numpy as np

from unmask.data.wild import WildConfig, list_tars, process_tar, run_wild_gates, sample_sheet


def _png(seed: int, size: int = 224) -> bytes:
    img = np.full((size, size, 3), seed % 251, dtype=np.uint8)  # flat colour encodes the frame
    return cv2.imencode(".png", img)[1].tobytes()


def _make_tar(path, tid: int, cls: str, seqs: dict[int, range], extra: list[str] = ()):
    with tarfile.open(path, "w") as tf:
        names = [(f"./{tid}/{cls}/{s}/{f}.png", s * 1000 + f) for s, fr in seqs.items() for f in fr]
        names += [(n, 0) for n in extra]
        for name, seed in reversed(names):  # arbitrary member order, as in the real tars
            data = _png(seed)
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_process_tar_groups_clips_and_picks_32(tmp_path):
    src = tmp_path / "src"
    (src / "fake_test").mkdir(parents=True)
    (src / "real_test").mkdir(parents=True)
    _make_tar(src / "fake_test" / "7.tar.gz", 7, "fake", {3: range(100, 160), 4: range(5)},
              extra=["./7/fake/../../evil.png", "./7/fake/x/1.png"])
    for i in range(8):
        _make_tar(src / "real_test" / f"{i + 1}.tar.gz", i + 1, "real", {1: range(40)})
        _make_tar(src / "fake_test" / f"{i + 10}.tar.gz", i + 10, "fake", {2: range(40)})
    tars = list_tars(src)
    assert tars[0][0].name == "7.tar.gz" and tars[0][1] == "FAKE" and tars[-1][1] == "REAL"

    out, cfg = tmp_path / "out", WildConfig()
    rows, st = process_tar(src / "fake_test" / "7.tar.gz", "FAKE", out, cfg)
    by = {r["video"]: r for r in rows}
    assert set(by) == {"fake_7_3", "fake_7_4"}
    assert by["fake_7_3"]["n_frames"] == 32 and by["fake_7_3"]["status"] == "ok"
    assert by["fake_7_4"]["n_frames"] == 5 and by["fake_7_4"]["status"] == "short"
    assert by["fake_7_3"]["source_tar"] == "fake_test/7.tar.gz" and by["fake_7_3"]["n_source_frames"] == 60
    assert st["skipped"] == 2 and st["read_ok"] == st["selected"] == 37
    assert not (tmp_path / "evil.png").exists() and not (out.parent / "evil.png").exists()
    # first and last selected frames are the first and last source frames (evenly spaced, numeric order)
    first = cv2.imread(str(out / "fake_7_3" / "f00.jpg"))
    last = cv2.imread(str(out / "fake_7_3" / "f31.jpg"))
    assert abs(first.mean() - (3100 % 251)) < 2 and abs(last.mean() - (3159 % 251)) < 2

    for p, lab in tars[1:]:
        r2, s2 = process_tar(p, lab, out, cfg)
        rows += r2
        st = {k: st[k] + s2[k] for k in st}
    assert sample_sheet(out, rows) is not None
    gates = run_wild_gates(out, rows, len(tars), [], st["selected"], st["read_ok"])
    assert all(g["ok"] for g in gates), gates
    gates = run_wild_gates(out, rows, len(tars) + 1, ["x: bad"], st["selected"], st["read_ok"])
    assert not gates[0]["ok"]
