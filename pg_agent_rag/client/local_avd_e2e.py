"""
Local AVD e2e helper — run on Windows only (not on AutoDL).

Uses adb for screenshot + tap when available; always calls remote RAG via RAG_URL.
Does not launch long android_world suites unless --run-tap is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from android_world_hook import inject_guidelines  # noqa: E402
from rag_client import RagClient  # noqa: E402


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def adb_ok() -> bool:
    p = run(["adb", "devices"])
    if p.returncode != 0:
        return False
    lines = [ln for ln in p.stdout.splitlines()[1:] if ln.strip() and "device" in ln]
    return bool(lines)


def adb_screenshot(path: Path) -> None:
    # Avoid exec-out issues on some Windows shells: screencap to device then pull
    remote = "/sdcard/pg_agent_rag_shot.png"
    r1 = run(["adb", "shell", "screencap", "-p", remote])
    if r1.returncode != 0:
        raise RuntimeError(r1.stderr or r1.stdout)
    r2 = run(["adb", "pull", remote, str(path)])
    if r2.returncode != 0:
        raise RuntimeError(r2.stderr or r2.stdout)


def adb_tap(x: int, y: int) -> None:
    r = run(["adb", "shell", "input", "tap", str(x), str(y)])
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-url", default=os.environ.get("RAG_URL", "http://127.0.0.1:18180"))
    parser.add_argument(
        "--summary",
        default="Android home screen with Settings, Chrome, and Clock icons.",
        help="Screen summary S_It (replace with MLLM output in real agent)",
    )
    parser.add_argument("--run-tap", action="store_true", help="Optionally tap center of screen via adb")
    parser.add_argument("--screenshot", action="store_true", help="Take one adb screenshot (saved under %%TEMP%%)")
    args = parser.parse_args()

    client = RagClient(base_url=args.rag_url, timeout=60)
    health = client.health()
    print("health:", json.dumps(health, ensure_ascii=False))
    if health.get("status") != "ok":
        raise SystemExit("RAG not healthy — start AutoDL server + SSH tunnel first")

    text, raw = inject_guidelines(args.summary, rag_url=args.rag_url)
    print(text)
    assert raw.get("guidelines"), "empty guidelines"

    if args.screenshot or args.run_tap:
        if not adb_ok():
            raise SystemExit("No adb device online. Start AVD / enable debugging first.")

    if args.screenshot:
        out = Path(tempfile.gettempdir()) / "pg_agent_rag_shot.png"
        adb_screenshot(out)
        print("screenshot:", out, "size=", out.stat().st_size)

    if args.run_tap:
        # Conservative demo tap near center of a typical 1080x1920 emulator
        print("adb tap 540 960 (demo; not from RAG coordinates)")
        adb_tap(540, 960)

    print("OK local e2e: RAG retrieve" + (" + adb" if (args.screenshot or args.run_tap) else " only"))


if __name__ == "__main__":
    main()
