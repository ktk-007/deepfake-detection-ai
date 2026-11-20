"""Download Report (PLAN 12): one self-contained HTML file, no external files, light and dark aware."""
import base64
import html
from datetime import datetime

COLORS = {"fake": "#c8102e", "real": "#15803d", "inconclusive": "#b45309"}
TITLES = {"fake": "Likely Fake", "real": "Likely Real", "inconclusive": "Inconclusive"}


def note_text(auc: dict) -> str:
    nums = []
    if "celebdf" in auc:
        nums.append(f"Celeb-DF v2 {auc['celebdf']:.3f}")
    if "wild" in auc:
        nums.append(f"WildDeepfake {auc['wild']:.3f}")
    measured = f" Measured AUC: {', '.join(nums)}." if nums else ""
    return ("Results are estimates and can be wrong on videos unlike the training data. Trained on DFDC face "
            f"swaps.{measured} Verify important findings with other evidence.")


def chart_svg(probs: list[float], low: float, high: float, w: int = 720, h: int = 200) -> str:
    pad_l, pad_b, pad_t = 40, 24, 10
    ph, pw = h - pad_b - pad_t, w - pad_l - 10
    n = max(1, len(probs))
    bw = pw / n
    y = lambda p: pad_t + ph * (1 - p)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Fake score per analysed frame">']
    for t in (0, 0.5, 1):
        parts.append(f'<text x="{pad_l - 6}" y="{y(t) + 4:.1f}" text-anchor="end" class="ax">{int(t * 100)}%</text>')
    parts.append(f'<rect x="{pad_l}" y="{y(high):.1f}" width="{pw}" height="{y(low) - y(high):.1f}" class="band"/>')
    for i, p in enumerate(probs):
        c = "fake" if p >= high else "real" if p <= low else "inconclusive"
        parts.append(f'<rect x="{pad_l + i * bw + bw * 0.15:.1f}" y="{y(p):.1f}" width="{bw * 0.7:.1f}" '
                     f'height="{ph * p:.1f}" fill="{COLORS[c]}" rx="2"/>')
    for t in (low, high):
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="thr"/>')
    parts.append(f'<line x1="{pad_l}" x2="{pad_l + pw}" y1="{y(0)}" y2="{y(0)}" class="base"/>')
    parts.append(f'<text x="{pad_l + pw / 2}" y="{h - 4}" text-anchor="middle" class="ax">Analysed frames 1-{n}'
                 '</text></svg>')
    return "".join(parts)


def build_report(res: dict, file_info: dict) -> str:
    e = html.escape
    lab = res["label"]
    show_fake = lab != "real"
    prob = res["score"] if show_fake else 1 - res["score"]
    conf = f'{res["confidence"]} confidence' if res.get("confidence") else ""
    faces = "".join(
        f'<figure><img alt="Face from frame {f["frame"]}" src="data:image/jpeg;base64,'
        f'{base64.b64encode(f["image"]).decode()}"><figcaption>frame {f["frame"]} · {f["prob"] * 100:.0f}%'
        '</figcaption></figure>' for f in res["faces"])
    band = (f'Band {res["low"]:.2f}-{res["high"]:.2f}'
            + ("" if res["calibrated"] else " (default band; calibration not fitted yet)"))
    info = " · ".join(e(str(v)) for v in (file_info.get("size"), file_info.get("resolution"),
                                          file_info.get("duration")) if v)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unmask report</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--line:#e6e6e3;--ink:#111;--muted:#6b6b6b;--band:rgba(180,83,9,.08)}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--card:#1b1b1b;--line:#2c2c2c;--ink:#f2f2f2;--muted:#a3a3a3;
--band:rgba(245,158,11,.12)}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:860px;margin:0 auto;padding:24px 16px}}
.logo{{font-weight:800;font-size:22px}} .card{{background:var(--card);border:1px solid var(--line);
border-radius:20px;padding:20px;margin:16px 0}}
.pill{{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;color:{COLORS[lab]};
background:color-mix(in srgb,{COLORS[lab]} 12%,transparent)}}
.big{{font-size:40px;font-weight:800;color:{COLORS[lab]}}} .muted{{color:var(--muted)}}
.bar{{height:10px;background:var(--line);border-radius:99px}} .bar>i{{display:block;height:100%;
border-radius:99px;background:{COLORS[lab]};width:{prob * 100:.1f}%}}
.faces{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px}}
figure{{margin:0}} img{{width:100%;border-radius:12px;display:block}} figcaption{{font-size:12px;color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .stats b{{display:block;font-size:17px}}
svg .ax{{font-size:11px;fill:var(--muted)}} svg .thr{{stroke:var(--muted);stroke-dasharray:4 4}}
svg .base{{stroke:var(--line)}} svg .band{{fill:var(--band)}}
</style></head><body><main>
<div class="logo">// Unmask</div>
<p class="muted">Report created {datetime.now():%Y-%m-%d %H:%M} · {e(file_info.get("name", "video"))}{" · " + info if info else ""}</p>
<section class="card"><span class="pill">{TITLES[lab]}</span> <span class="muted">{e(conf)}</span>
<p>{e(res["sentence"])}</p>
<div class="muted">{"Fake" if show_fake else "Real"} probability</div><div class="big">{prob * 100:.1f}%</div>
<div class="bar"><i></i></div><p class="muted">{e(band)}</p></section>
<section class="card"><h2>Frame-by-frame analysis</h2>{chart_svg(res["frame_probs"], res["low"], res["high"])}</section>
<section class="card"><h2>Faces analysed</h2><div class="faces">{faces}</div>
<p class="muted">Highlighted areas influenced the result most.</p>
<div class="stats"><div><span class="muted">Model Used</span><b>{e(res["model"])}</b></div>
<div><span class="muted">Frames Analysed</span><b>{res["n_frames"]}</b></div>
<div><span class="muted">Processing Time</span><b>{res["seconds"]:.1f} s</b></div></div></section>
<p class="muted">{e(note_text(res.get("auc", {})))}</p>
</main></body></html>"""
