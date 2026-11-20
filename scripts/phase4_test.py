"""Phase 4, step 2 (PLAN 11): score each final model ONCE on each test set and write the final table.

Requires results\\phase4\\choices.json from phase4_val.py (every choice already fixed on part 08). For each model
(E7 first) and each test set (DFDC part 09, Celeb-DF v2, WildDeepfake) the frame scores are saved the first time;
a set that is already scored is never re-scored (results\\phase4\\test_ledger.json records when each was opened).
Metrics use the model's part-08 temperature and band: AUC with a 95% bootstrap interval (by original-video group on
DFDC, by video on Celeb-DF, by source tar on WildDeepfake), AP, EER, log loss, ECE, balanced accuracy, share of
videos in the Inconclusive band. Writes results\\final_table.csv / .md, results\\phase4\\metrics.json,
results\\app_numbers.json (Celeb-DF / WildDeepfake AUC of the app's default model) and ROC curves as SVG.

  python scripts\\phase4_test.py
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unmask.common import CROPS_DIR, PROJECT_ROOT, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402
from unmask.eval.phase4 import (SET_LABELS, TEST_SETS, Scores, bootstrap_auc, full_metrics, load_set,  # noqa: E402
                                score_videos)

RESULTS = PROJECT_ROOT / "results"
ORDER = ["e7_l14", "e2_b16", "e3_effb4", "e1_linear_probe", "e8_gend"]
COLORS = {"e7_l14": "#1F4E9C", "e2_b16": "#6A8FD1", "e3_effb4": "#B91C1C", "e1_linear_probe": "#9AA5B8",
          "e8_gend": "#B26A00", "ensemble": "#111111"}


def roc_svg(curves: list, title: str) -> str:
    w, h, pad = 420, 420, 46
    pw = w - pad - 12
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 90}" font-family="Arial" font-size="11">',
             f'<rect width="{w}" height="{h + 90}" fill="white"/>',
             f'<text x="{w / 2}" y="16" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>',
             f'<rect x="{pad}" y="24" width="{pw}" height="{pw}" fill="none" stroke="#C9D3E3"/>',
             f'<line x1="{pad}" y1="{24 + pw}" x2="{pad + pw}" y2="24" stroke="#C9D3E3" stroke-dasharray="4 4"/>',
             f'<text x="{pad + pw / 2}" y="{24 + pw + 28}" text-anchor="middle">False positive rate</text>',
             f'<text x="12" y="{24 + pw / 2}" transform="rotate(-90 12 {24 + pw / 2})" text-anchor="middle">True positive rate</text>']
    for k, (name, fpr, tpr, auc, color) in enumerate(curves):
        pts = " ".join(f"{pad + f * pw:.1f},{24 + (1 - t) * pw:.1f}" for f, t in zip(fpr, tpr))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{pad}" y="{24 + pw + 42 + 14 * k - 8}" width="10" height="3" fill="{color}"/>'
                     f'<text x="{pad + 16}" y="{24 + pw + 42 + 14 * k}">{name}: AUC {auc:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    out = args.results / "phase4"
    log, log_path = setup_logger("phase4_test")
    t0 = time.time()
    lock_path = out / "choices.json"
    if not lock_path.exists():
        raise SystemExit("results\\phase4\\choices.json not found: run scripts\\phase4_val.py first (choices on part 08)")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    choice = lock["models"]
    ledger_path = out / "test_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []
    log.info(f"choices locked at {lock['locked_at']} on part 08; scoring each test set once per model")

    from unmask.train.loop import load_model_checkpoint
    sets = {name: load_set(name, args.crops_dir) for name in TEST_SETS}
    for name, vids in sets.items():
        log.info(f"{SET_LABELS[name]}: {len(vids)} videos ({sum(v.label for v in vids)} fake)")

    runs = [r for r in ORDER if r in choice]
    scores = {}
    for run in runs:
        c = choice[run]
        model = None
        for name, vids in sets.items():
            for flip in ([False, True] if c["flip"] else [False]):
                path = out / f"test_{run}_{name}{'_flip' if flip else ''}.npz"
                if not path.exists():
                    if model is None:
                        model = load_model_checkpoint(Path(c["ckpt"]), args.device)[0]
                        log.info(f"{run}: loaded {c['ckpt']}")
                    n = sum(len(v.frames) for v in vids)
                    bar = tqdm(total=-(-n // args.batch), desc=f"{run} {name}{' flip' if flip else ''}", unit="batch",
                               dynamic_ncols=True)
                    s = score_videos(model, vids, args.device, flip=flip, progress=bar, batch=args.batch)
                    bar.close()
                    s.save(path, run=run, set=name, flip=flip)
                    ledger.append(dict(run=run, set=name, flip=flip, opened=datetime.now().isoformat(timespec="seconds")))
                    ledger_path.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
                scores[(run, name, flip)] = Scores.load(path)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    def final_scores(run, name):
        s = scores[(run, name, False)]
        if choice[run]["flip"]:
            s = Scores(s.keys, s.labels, s.groups, s.offsets, (s.z + scores[(run, name, True)].z) / 2)
        return s

    metrics, rows = {}, []
    for run in runs:
        c = choice[run]
        metrics[run] = {name: full_metrics(final_scores(run, name), c["temperature"], c["band"], args.boot)
                        for name in TEST_SETS}
    ens = lock["ensemble"]
    if ens.get("adopted"):
        metrics["ensemble"] = {}
        for name in TEST_SETS:
            mem = ens["members"]
            probs = np.mean([final_scores(r, name).video_probs(choice[r]["temperature"]) for r in mem], axis=0)
            base = final_scores(mem[0], name)
            pc = np.clip(probs, 1e-6, 1 - 1e-6)    # one 'frame' per video holding the ensemble's logit
            p_scores = Scores(base.keys, base.labels, base.groups, np.arange(len(pc) + 1), np.log(pc / (1 - pc)))
            metrics["ensemble"][name] = full_metrics(p_scores, 1.0, ens["band"], args.boot)

    labels = {r: choice[r]["label"] for r in runs} | {"ensemble": "E10 Ensemble (" + " + ".join(ens.get("members", [])) + ")"}
    for run, m in metrics.items():
        c = choice.get(run, {})
        row = dict(model=labels[run], options=("flip " if c.get("flip") else "") + ("" if run != "ensemble" else "calibrated mean"),
                   part08_clean=c.get("val_clean32_auc", ""), part08_val_robust=c.get("val_robust", ens.get("val_robust")))
        for name in TEST_SETS:
            row[f"{name}_auc"] = m[name]["auc"]
            row[f"{name}_ci"] = f"{m[name]['auc_lo']:.3f}-{m[name]['auc_hi']:.3f}"
        rows.append(row)

    args.results.mkdir(parents=True, exist_ok=True)
    with open(args.results / "final_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["| Model | Part 08 clean | Part 08 val_robust | DFDC part 09 | Celeb-DF v2 | WildDeepfake |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['model']} | {fmt(r['part08_clean'])} | {fmt(r['part08_val_robust'])} | "
                  + " | ".join(f"{r[f'{n}_auc']:.4f} ({r[f'{n}_ci']})" for n in TEST_SETS) + " |")
    (args.results / "final_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    app_model = "ensemble" if ens.get("adopted") else ("e7_l14" if "e7_l14" in metrics else runs[0])
    per_key = {(choice[r]["key"] if r != "ensemble" else "ensemble"): dict(celebdf=metrics[r]["celebdf"]["auc"],
                                                                          wild=metrics[r]["wild"]["auc"]) for r in metrics}
    (args.results / "app_numbers.json").write_text(json.dumps(dict(
        model=labels[app_model], celebdf=metrics[app_model]["celebdf"]["auc"], wild=metrics[app_model]["wild"]["auc"],
        models=per_key), indent=1), encoding="utf-8")

    plots = args.results / "plots"
    plots.mkdir(exist_ok=True)
    for name in TEST_SETS:
        curves = []
        for run in list(runs) + (["ensemble"] if ens.get("adopted") else []):
            if run == "ensemble":
                p = np.mean([final_scores(r, name).video_probs(choice[r]["temperature"]) for r in ens["members"]], axis=0)
                y = final_scores(ens["members"][0], name).labels
            else:
                s = final_scores(run, name)
                p, y = s.video_probs(choice[run]["temperature"]), s.labels
            fpr, tpr, _ = roc_curve(y, p)
            curves.append((labels[run], fpr, tpr, metrics[run][name]["auc"], COLORS.get(run, "#333")))
        (plots / f"roc_{name}.svg").write_text(roc_svg(curves, f"ROC: {SET_LABELS[name]}"), encoding="utf-8")

    log.info("")
    log.info("==== PHASE 4 TEST SUMMARY ====")
    for r in rows:
        log.info(f"{r['model']}: part 09 {r['test09_auc']:.4f} ({r['test09_ci']}) | Celeb-DF {r['celebdf_auc']:.4f} "
                 f"({r['celebdf_ci']}) | WildDeepfake {r['wild_auc']:.4f} ({r['wild_ci']})")
    log.info(f"table: {args.results / 'final_table.md'} | plots: {plots} | app numbers from {labels[app_model]}")
    log.info(f"time {(time.time() - t0) / 60:.1f} min | free disk {free_gb(PROJECT_ROOT):.1f} GB | log {log_path}")
    log.info("==============================")
    return metrics


if __name__ == "__main__":
    main()
