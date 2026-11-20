"""Start the Unmask app (PLAN 12) on this computer only, then open http://127.0.0.1:8000.

  python scripts\\run_app.py
  python scripts\\run_app.py --device cpu        (no GPU; slower)

Uses the GPU by default: do not run it while a training, extraction or evaluation job is running.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="default: cuda if available")
    args = ap.parse_args()

    import uvicorn

    from unmask.app.pipeline import Analyzer
    from unmask.app.server import create_app

    an = Analyzer(device=args.device)
    found = an.models()
    print(f"Unmask: device {an.device}; models: {', '.join(m.label for m in found) or 'none found'}")
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(an), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
