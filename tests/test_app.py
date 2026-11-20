"""Unmask app (PLAN 12): pipeline, verdict rules, server endpoints and report. CPU and the tiny test model only."""
import json
import time

import cv2
import numpy as np
import pytest

from unmask.app.pipeline import (AnalysisError, Analyzer, find_models, load_calibration, measured_auc,
                                 verdict)
from unmask.app.report import build_report, chart_svg, note_text
from unmask.models.build import TinyNet


def _video(path, n=40, seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n):
        vw.write(np.roll(base, i, axis=1).copy())
    vw.release()
    return path


def stub_detector(frames):
    return [[110.0, 70.0, 210.0, 170.0] for _ in frames]


@pytest.fixture
def setup(tmp_path):
    runs, weights, results = tmp_path / "runs", tmp_path / "weights", tmp_path / "results"
    (runs / "e2_b16").mkdir(parents=True)
    (runs / "e2_b16" / "best.pt").write_bytes(b"x")
    weights.mkdir()
    results.mkdir()
    an = Analyzer(device="cpu", detector=stub_detector, loader=lambda e: TinyNet().eval(), runs=runs,
                  weights=weights, results=results)
    return an, tmp_path


def test_verdict_bands_and_confidence():
    p = np.full(32, 0.97)
    assert verdict(0.97, p, 0.35, 0.65) == dict(
        label="fake", confidence="High", agree=1.0,
        sentence="Signs of face manipulation were found in most analysed frames.")
    assert verdict(0.70, p, 0.35, 0.65)["confidence"] == "Medium"           # outside the band by < 0.15
    mixed = np.array([0.9] * 20 + [0.2] * 12)                                # 62% of frames agree
    assert verdict(0.9, mixed, 0.35, 0.65)["confidence"] == "Medium"
    r = verdict(0.05, np.full(32, 0.05), 0.35, 0.65)
    assert r["label"] == "real" and r["confidence"] == "High"
    assert r["sentence"] == "No significant signs of face manipulation were found."
    u = verdict(0.5, p, 0.35, 0.65)
    assert u["label"] == "inconclusive" and u["confidence"] is None
    assert u["sentence"] == "The model is unsure. Some frames look manipulated and others do not."


def test_model_discovery_order_weights_first_and_ensemble(tmp_path):
    runs, weights, results = tmp_path / "r", tmp_path / "w", tmp_path / "res"
    for r in ("e7_l14", "a7b_ce_lr3e-4", "e3_effb4"):
        (runs / r).mkdir(parents=True)
        (runs / r / "best.pt").write_bytes(b"x")
    (runs / "e7_l14" / "avg.pt").write_bytes(b"x")
    (runs / "e7_l14" / "avg_decision.json").write_text(json.dumps({"use_avg": True}))
    weights.mkdir()
    (weights / "vit_b16.pt").write_bytes(b"x")
    results.mkdir()
    (results / "ensemble.json").write_text(json.dumps({"adopted": True, "members": ["vit_l14", "effb4"]}))
    found = find_models(runs, weights, results)
    assert [m.key for m in found] == ["vit_l14", "vit_b16", "effb4", "ensemble"]
    assert found[0].path.name == "avg.pt" and found[1].path == weights / "vit_b16.pt"
    assert found[3].members == ["vit_l14", "effb4"] and found[3].label == "Ensemble (ViT-L/14 + EfficientNet)"


def test_calibration_and_numbers_fallbacks(tmp_path):
    assert load_calibration("vit_b16", tmp_path) == dict(temperature=1.0, low=0.35, high=0.65, fitted=False)
    assert measured_auc(tmp_path) == {}
    (tmp_path / "calibration.json").write_text(json.dumps({"vit_b16": {"temperature": 1.4, "low": 0.3, "high": 0.7}}))
    (tmp_path / "app_numbers.json").write_text(json.dumps({"celebdf": 0.81234, "wild": 0.7}))
    assert load_calibration("vit_b16", tmp_path)["fitted"] and load_calibration("vit_b16", tmp_path)["high"] == 0.7
    n = note_text(measured_auc(tmp_path))
    assert "Celeb-DF v2 0.812" in n and "WildDeepfake 0.700" in n
    assert "Measured AUC" not in note_text({})                 # never a number that results\ did not produce


def test_pipeline_end_to_end(setup):
    an, tmp = setup
    steps = []
    res = an.analyze(_video(tmp / "v.mp4"), "vit_b16", progress=lambda s, f: steps.append(s))
    assert steps[0] == "extract" and steps[-1] == "result" and "analyze" in steps
    assert res["n_frames"] == 32 and len(res["frame_probs"]) == 32
    assert 0 <= res["score"] <= 1 and res["label"] in ("fake", "real", "inconclusive")
    assert len(res["faces"]) == 6 and all(f["image"][:2] == b"\xff\xd8" for f in res["faces"])
    img = cv2.imdecode(np.frombuffer(res["faces"][0]["image"], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (224, 224, 3)
    assert res["model"] == "CLIP ViT-B/16 (faster)" and res["calibrated"] is False


def test_no_face_and_unreadable(setup):
    an, tmp = setup
    an._detector = lambda frames: None
    with pytest.raises(AnalysisError) as e:
        an.analyze(_video(tmp / "v.mp4"), "vit_b16")
    assert e.value.kind == "no_face"
    bad = tmp / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(AnalysisError) as e:
        an.analyze(bad, "vit_b16")
    assert e.value.kind == "unreadable"


def test_report_is_self_contained():
    res = dict(label="fake", confidence="High", sentence="s", score=0.9, frame_probs=[0.9, 0.2, 0.5], low=0.35,
               high=0.65, calibrated=False, faces=[dict(frame=1, prob=0.9, image=b"\xff\xd8abc")], model="M",
               n_frames=3, seconds=1.2, auc={})
    page = build_report(res, {"name": "<clip>.mp4", "size": "1 MB"})
    assert page.startswith("<!doctype html>") and "&lt;clip&gt;" in page and "<clip>" not in page
    assert "data:image/jpeg;base64," in page and "src=\"http" not in page and "Likely Fake" in page
    assert chart_svg([0.1, 0.9], 0.35, 0.65).count("<rect") == 3     # band + 2 bars


def test_server_upload_poll_result_report(setup):
    from fastapi.testclient import TestClient
    from unmask.app.server import create_app
    an, tmp = setup
    client = TestClient(create_app(an))
    assert client.get("/").status_code == 200 and "Deepfake Detection" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/models").json() == [{"key": "vit_b16", "label": "CLIP ViT-B/16 (faster)"}]

    vid = _video(tmp / "clip.mp4").read_bytes()
    r = client.post("/api/analyze", files={"file": ("clip.mp4", vid, "video/mp4")},
                    data={"model": "vit_b16", "size": "1 MB", "resolution": "320 x 240", "duration": "0:04"})
    assert r.status_code == 200
    job = r.json()["job"]
    for _ in range(200):
        j = client.get(f"/api/jobs/{job}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert j["result"]["faces"][0]["image"].startswith("data:image/jpeg;base64,")
    assert "Results are estimates" in j["result"]["note"]
    rep = client.get(f"/api/report/{job}")
    assert rep.status_code == 200 and "attachment" in rep.headers["content-disposition"]
    assert "unmask_report_clip.html" in rep.headers["content-disposition"]

    assert client.post("/api/analyze", files={"file": ("x.txt", b"hi", "text/plain")},
                       data={"model": "vit_b16"}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("x.mp4", vid, "video/mp4")},
                       data={"model": "nope"}).status_code == 400
    assert client.get("/api/jobs/missing").status_code == 404


def test_server_reports_no_face(setup):
    from fastapi.testclient import TestClient
    from unmask.app.server import create_app
    an, tmp = setup
    an._detector = lambda frames: None
    client = TestClient(create_app(an))
    job = client.post("/api/analyze", files={"file": ("c.mp4", _video(tmp / "c.mp4").read_bytes(), "video/mp4")},
                      data={"model": "vit_b16"}).json()["job"]
    for _ in range(200):
        j = client.get(f"/api/jobs/{job}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "error" and j["error"]["kind"] == "no_face"
