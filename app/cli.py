"""
CLI entry point.

After `pip install efficient-tokenizer`, users run:

    efficient-tokenizer serve            # start on :8000
    efficient-tokenizer serve --port 9000
    efficient-tokenizer serve --dry-run  # no real LLM calls
    efficient-tokenizer test             # run smoke tests
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys


def _serve(port: int, dry_run: bool, workers: int, reload: bool) -> None:
    if dry_run:
        os.environ["DISPATCH_DRY_RUN"] = "true"
        print("[efficient-tokenizer] DRY RUN mode — no LLM calls will be made.")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
    else:
        cmd += ["--workers", str(workers)]

    print(f"[efficient-tokenizer] Starting proxy on http://0.0.0.0:{port}")
    print(f"[efficient-tokenizer] Admin dashboard: http://localhost:{port}/dashboard")
    print(f"[efficient-tokenizer] Health check:    http://localhost:{port}/health")
    print(f"[efficient-tokenizer] Docs:            http://localhost:{port}/docs")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[efficient-tokenizer] Stopped.")


def _test() -> None:
    os.environ.setdefault("DISPATCH_DRY_RUN", "true")
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_test.py")
    script = os.path.abspath(script)
    result = subprocess.run([sys.executable, script])
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="efficient-tokenizer",
        description="Drop-in LLM proxy with token compression and observability.",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_p = sub.add_parser("serve", help="Start the proxy server")
    serve_p.add_argument("--port",     type=int,  default=8000, help="Port to listen on (default: 8000)")
    serve_p.add_argument("--workers",  type=int,  default=2,    help="Number of worker processes (default: 2)")
    serve_p.add_argument("--reload",   action="store_true",     help="Enable auto-reload (dev mode)")
    serve_p.add_argument("--dry-run",  action="store_true",     help="Skip actual LLM API calls")

    # test
    sub.add_parser("test", help="Run smoke tests (no API key required)")

    args = parser.parse_args()

    if args.command == "serve":
        _serve(
            port=args.port,
            dry_run=args.dry_run,
            workers=args.workers,
            reload=args.reload,
        )
    elif args.command == "test":
        _test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
