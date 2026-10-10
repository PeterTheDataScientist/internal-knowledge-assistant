"""Builds the demo clip and poster from the Streamlit app running locally.

    python scripts/make_demo_clip.py --out <folder>
    python make_demo_clip.py --fetch <folder> --out <folder>   # outside a checkout

It starts streamlit_app.py on localhost, drives it with Playwright through six states
(landing page, a cited answer, the opened source, a two-document answer, an injection
refusal, an out-of-scope refusal), screenshots each one, and joins the stills into a
16.2 second H.264 clip with short crossfades: knowledge-assistant-tour.mp4 (1024x576,
10 fps) and knowledge-assistant-tour-poster.jpg (the cited answer).

With --fetch it first downloads the files the demo needs from the public repository, so it
also runs where only raw.githubusercontent.com is reachable. Needs streamlit, playwright,
pillow and ffmpeg (or imageio-ffmpeg). Set CHROMIUM_PATH to use a preinstalled Chromium.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

RAW = "https://raw.githubusercontent.com/PeterTheDataScientist/internal-knowledge-assistant/main/"
DEMO_FILES = [
    "requirements.txt", "streamlit_app.py", ".streamlit/config.toml", "app/__init__.py",
    "app/api/__init__.py", "app/api/routes.py", "app/config.py", "app/embeddings/__init__.py",
    "app/embeddings/base.py", "app/embeddings/providers.py", "app/errors.py",
    "app/indexing/__init__.py", "app/indexing/lexical_index.py", "app/indexing/vector_store.py",
    "app/ingestion/__init__.py", "app/ingestion/chunker.py", "app/ingestion/loader.py",
    "app/llm/__init__.py", "app/llm/base.py", "app/llm/factory.py", "app/llm/groq_provider.py",
    "app/llm/mock_provider.py", "app/main.py", "app/observability/__init__.py",
    "app/observability/logging_config.py", "app/observability/metrics.py",
    "app/observability/middleware.py", "app/rag/__init__.py", "app/rag/grounding.py",
    "app/rag/knowledge_base.py", "app/rag/memory.py", "app/rag/pipeline.py", "app/rag/prompts.py",
    "app/retrieval/__init__.py", "app/retrieval/retriever.py", "app/safety/__init__.py",
    "app/safety/guards.py", "app/safety/patterns.py", "app/schemas.py",
    "data/00-README-CORPUS.md", "data/aml-kyc-policy.md", "data/aml-sanctions-screening.md",
    "data/credit-mortgage-policy.md", "data/credit-policy.md", "data/data-classification-policy.md",
    "data/faq-employees.md", "data/group-overview.md", "data/hr-code-of-conduct.md",
    "data/hr-leave-policy.md", "data/information-security-policy.md", "data/product-accounts.md",
    "data/product-cards.md", "data/product-digital-banking.md", "data/product-funeral-plan.md",
    "data/sop-account-opening.md", "data/sop-cash-handling.md", "data/staff-notices.md",
]

VIEW = {"width": 1536, "height": 674}
CROP = (304, 0, 1502, 674)  # the main column at 16:9, without the sidebar edge or scrollbar
SIZE = (1024, 576)
HOLDS = [2.6, 3.2, 3.0, 3.2, 3.2, 3.0]  # seconds each still stays on screen
FADE = 0.4
REQUEST_ID = r"/Request id: ([0-9a-f]+)/"


def fetch(dest: Path) -> Path:
    """Downloads the demo's files from the public repository into dest."""
    for name in DEMO_FILES:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + name, timeout=30) as response:
            target.write_bytes(response.read())
    return dest


def ffmpeg_binary() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # optional fallback: pip install imageio-ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"the app did not start at {url}")


def capture(url: str, shots: Path) -> list[Path]:
    """Walks the app through the six states and screenshots each one."""
    paths: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport=VIEW, device_scale_factor=1)
        page.goto(url, wait_until="networkidle", timeout=120_000)
        page.get_by_text("Try an example").wait_for(timeout=120_000)
        page.add_style_tag(
            content="[data-testid='stToolbar'],[data-testid='stDecoration'],[role='tooltip'],"
            "[data-baseweb='tooltip']{display:none!important}"
        )
        page.wait_for_timeout(1500)

        def shot(name: str) -> None:
            page.mouse.move(1530, 668)
            page.wait_for_timeout(400)
            path = shots / f"{len(paths) + 1}-{name}.png"
            page.screenshot(path=str(path))
            paths.append(path)

        def scroll(delta: int) -> None:
            page.mouse.move(900, 400)
            page.mouse.wheel(0, delta)
            page.wait_for_timeout(700)

        def ask(label: str) -> None:
            before = page.evaluate(
                f"() => {{ const m = document.body.innerText.match({REQUEST_ID}); return m ? m[1] : ''; }}"
            )
            page.get_by_role("button", name=label, exact=True).click()
            page.wait_for_function(
                f"(b) => {{ const m = document.body.innerText.match({REQUEST_ID});"
                " return m && m[1] !== b && !document.querySelector('[data-testid=\"stSkeleton\"]'); }",
                arg=before,
                timeout=60_000,
            )
            page.wait_for_timeout(800)

        shot("landing")
        ask("Annual leave")
        scroll(400)
        shot("answer")
        page.locator("[data-testid='stExpander'] summary").first.click()
        page.wait_for_timeout(800)
        scroll(300)
        shot("source")
        for label, name in [
            ("Two documents", "two-documents"),
            ("Injection attempt", "injection"),
            ("Not in the documents", "not-in-documents"),
        ]:
            scroll(-5000)
            ask(label)
            scroll(400)
            shot(name)
        browser.close()
    return paths


def build(stills: list[Path], out: Path, work: Path) -> None:
    """Crops the stills to the main column and joins them with crossfades."""
    frames = []
    for index, still in enumerate(stills, 1):
        frame = work / f"frame{index}.png"
        Image.open(still).convert("RGB").crop(CROP).resize(SIZE, Image.LANCZOS).save(frame)
        frames.append(frame)
    inputs: list[str] = []
    for frame, hold in zip(frames, HOLDS):
        inputs += ["-loop", "1", "-t", str(hold), "-i", str(frame)]
    chains = [f"[{i}:v]fps=10,format=yuv420p,setsar=1[v{i}]" for i in range(len(frames))]
    offset, last = 0.0, "v0"
    for i in range(1, len(frames)):
        offset += HOLDS[i - 1] - FADE
        chains.append(f"[{last}][v{i}]xfade=transition=fade:duration={FADE}:offset={offset:.1f}[a{i}]")
        last = f"a{i}"
    graph = ";".join(chains) + f";[{last}]format=yuv420p[out]"
    subprocess.run(
        [ffmpeg_binary(), "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
         "-map", "[out]", "-c:v", "libx264", "-preset", "veryslow", "-tune", "stillimage",
         "-crf", "30", "-g", "300", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-r", "10", str(out / "knowledge-assistant-tour.mp4")],
        check=True,
    )
    Image.open(frames[1]).save(out / "knowledge-assistant-tour-poster.jpg", quality=85, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="folder for the clip and poster")
    parser.add_argument("--fetch", type=Path, help="download the demo's files here first")
    args = parser.parse_args()
    app_dir = fetch(args.fetch) if args.fetch else Path(__file__).resolve().parent.parent
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_frames"
    work.mkdir(exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=app_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        wait_for(url)
        build(capture(url, work), args.out, work)
    finally:
        server.terminate()
    shutil.rmtree(work, ignore_errors=True)
    for name in ("knowledge-assistant-tour.mp4", "knowledge-assistant-tour-poster.jpg"):
        print(name, (args.out / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
