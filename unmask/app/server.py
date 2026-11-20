"""FastAPI backend for the Unmask page (PLAN 12). Local only (127.0.0.1).

  GET  /                 the page (static/index.html)
  GET  /api/models       models found now: [{key, label}]
  POST /api/analyze      multipart: file, model -> {job}
  GET  /api/jobs/{job}   {status, step, progress, result | error}
  GET  /api/report/{job} the self-contained HTML report (download)

The upload goes to a temp file that is deleted as soon as the frames are read; the analysis runs in one
background thread at a time (one GPU job). Results stay in memory for the last few jobs only (for the report).
"""
import base64
import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import STEPS, AnalysisError, Analyzer
from .report import build_report, note_text

STATIC = Path(__file__).resolve().parent / "static"
ALLOWED = {".mp4", ".mov", ".avi"}
MAX_BYTES = 100 * 1024 * 1024
KEEP_JOBS = 8


def create_app(analyzer: Analyzer | None = None) -> FastAPI:
    app = FastAPI(title="Unmask", docs_url=None, redoc_url=None)
    an = analyzer or Analyzer()
    jobs: "OrderedDict[str, dict]" = OrderedDict()
    gpu = threading.Lock()

    def run(job_id: str, path: Path, model_key: str):
        job = jobs[job_id]

        def progress(step, frac):
            job.update(status="running", step=step, progress=round(float(frac), 3))

        try:
            with gpu:
                job.update(status="running", step=STEPS[0], progress=0.0)
                res = an.analyze(path, model_key, progress=progress)
            job.update(status="done", step="result", progress=1.0, result=res)
        except AnalysisError as e:
            job.update(status="error", error={"kind": e.kind, "message": str(e)})
        except Exception as e:                      # anything else: the generic message, details in the console
            print(f"[unmask] analysis failed: {type(e).__name__}: {e}")
            job.update(status="error", error={"kind": "unreadable", "message": "This file can't be analyzed."})
        finally:
            path.unlink(missing_ok=True)            # the video is never kept

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def models():
        return [m.public() for m in an.models()]

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), model: str = Form(...), size: str = Form(""),
                      resolution: str = Form(""), duration: str = Form("")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(400, "This file can't be analyzed. Use MP4, MOV or AVI.")
        if model not in {m.key for m in an.models()}:
            raise HTTPException(400, f"Model {model!r} is not available.")
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="unmask_upload_")
        path, total = Path(tmp), 0
        try:
            with open(fd, "wb") as out:
                while chunk := await file.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(413, "This file is larger than 100 MB.")
                    out.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = dict(status="queued", step=STEPS[0], progress=0.0,
                            file=dict(name=file.filename, size=size, resolution=resolution, duration=duration))
        while len(jobs) > KEEP_JOBS:
            jobs.popitem(last=False)
        threading.Thread(target=run, args=(job_id, path, model), daemon=True).start()
        return {"job": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        out = {k: j[k] for k in ("status", "step", "progress") if k in j}
        if "error" in j:
            out["error"] = j["error"]
        if "result" in j:
            r = dict(j["result"])
            r["faces"] = [dict(f, image="data:image/jpeg;base64," + base64.b64encode(f["image"]).decode())
                          for f in r["faces"]]
            r["note"] = note_text(r.get("auc", {}))
            out["result"] = r
        return JSONResponse(out)

    @app.get("/api/report/{job_id}")
    def report(job_id: str):
        j = jobs.get(job_id)
        if j is None or "result" not in j:
            raise HTTPException(404, "no finished analysis with this id")
        name = "".join(c for c in Path(j["file"].get("name") or "video").stem if c.isascii() and
                       (c.isalnum() or c in "-_"))[:60] or "video"
        return HTMLResponse(build_report(j["result"], j["file"]),
                            headers={"Content-Disposition": f'attachment; filename="unmask_report_{name}.html"'})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
