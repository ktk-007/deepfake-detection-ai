"""Gate 5 (PLAN 12): run the app's own analysis pipeline on the held-out demo videos and time each one.

Checks the app's default model (first in the dropdown) and, if adopted, the ensemble, with the
calibration in results\\. Each video must finish in under 20 seconds. Writes results\\app_check.json and one
downloadable HTML report per video and model in results\\app_reports\\. Run only when no other GPU job runs.

  python scripts\\app_check.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmask.app.pipeline import Analyzer, AnalysisError  # noqa: E402
from unmask.app.report import build_report  # noqa: E402
from unmask.common import PROJECT_ROOT, setup_logger  # noqa: E402

DEMO = PROJECT_ROOT / "data" / "app_test"
VIDEOS = [("enyfzkpdeq.mp4", "real", "DFDC 18"), ("eyfqtmmjhg.mp4", "fake", "DFDC 18"), ("lbqrludiqd.mp4", "fake", "DFDC 18"),
          ("yirkaxfwgb.mp4", "real", "DFDC 06"), ("lwasdlhiiv.mp4", "fake", "DFDC 06"),
          ("celebdf/id45_0003.mp4", "real", "Celeb-DF"), ("celebdf/id48_id41_0001.mp4", "fake", "Celeb-DF")]
VERDICT = {"fake": "Likely Fake", "real": "Likely Real", "inconclusive": "Inconclusive"}


def run_model(an, entry, log, out_dir):
    an.analyze(DEMO / VIDEOS[0][0], entry.key)          # warm-up: loads the model(s) and the face detector once
    rows = []
    for name, truth, source in VIDEOS:
        t0 = time.time()
        try:
            r = an.analyze(DEMO / name, entry.key)
            secs = time.time() - t0
            (out_dir / f"{entry.key}_{Path(name).stem}.html").write_text(build_report(r, {"name": Path(name).name}),
                                                                        encoding="utf-8")
            rows.append(dict(name=Path(name).stem, source=source, truth=truth, label=r["label"], verdict=VERDICT[r["label"]],
                             confidence=r["confidence"], score=round(r["score"], 4), seconds=round(secs, 2),
                             calibrated=r["calibrated"], correct=(r["label"] == truth)))
        except AnalysisError as e:
            rows.append(dict(name=Path(name).stem, source=source, truth=truth, label="error", verdict=str(e),
                             confidence=None, score=None, seconds=round(time.time() - t0, 2), correct=False))
        log.info(f"[{entry.key}] {rows[-1]['name']} ({truth}): {rows[-1]['verdict']} {rows[-1]['confidence'] or ''} "
                 f"score {rows[-1]['score']} | {rows[-1]['seconds']:.1f} s")
    return dict(model=entry.label, model_key=entry.key, videos=rows, all_under_20s=all(r["seconds"] < 20 for r in rows),
                correct=sum(r["correct"] for r in rows))


def main():
    log, log_path = setup_logger("app_check")
    an = Analyzer()
    models = an.models()
    if not models:
        raise SystemExit("no model found in weights\\ or runs\\")
    log.info(f"device {an.device}; models: {', '.join(m.label for m in models)}")
    out_dir = PROJECT_ROOT / "results" / "app_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checked = [models[0]] + [m for m in models if m.key == "ensemble"]
    res = [run_model(an, m, log, out_dir) for m in checked]
    out = dict(res[0], device=an.device, checks=res)
    (PROJECT_ROOT / "results" / "app_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("")
    log.info("==== APP CHECK SUMMARY ====")
    for r in res:
        log.info(f"{r['model']}: {r['correct']}/{len(r['videos'])} verdicts match the truth (Inconclusive counts as "
                 f"not matching) | slowest {max(v['seconds'] for v in r['videos']):.1f} s | all under 20 s: {r['all_under_20s']}")
    log.info(f"reports in {out_dir} | log {log_path}")
    log.info("===========================")


if __name__ == "__main__":
    main()
