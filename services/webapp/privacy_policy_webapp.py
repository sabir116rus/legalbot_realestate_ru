"""Minimal Telegram Web App that renders the privacy policy text."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

from aiohttp import web


_PACKAGE_DIR = Path(__file__).resolve().parent
_POLICY_HTML_PATH = _PACKAGE_DIR / "privacy_policy.html"


def get_privacy_policy_html() -> str:
    """Return the raw HTML for the privacy policy page."""
    return _POLICY_HTML_PATH.read_text(encoding="utf-8")


def create_app() -> web.Application:
    """Create an aiohttp application serving the privacy policy page."""
    html_content = get_privacy_policy_html()

    async def handle(_: web.Request) -> web.Response:
        return web.Response(text=html_content, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", handle)
    app.router.add_get("/privacy-policy", handle)
    return app


async def _run_app(host: str, port: int) -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    print(  # noqa: T201 - Utility script output
        f"Privacy policy web app is running on http://{host}:{port}/privacy-policy",
        flush=True,
    )

    stop_event = asyncio.Event()

    try:
        await stop_event.wait()
    except asyncio.CancelledError:  # pragma: no cover - graceful shutdown path
        pass
    finally:
        await runner.cleanup()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the LegalBot privacy policy web app")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    default_port = int(os.getenv("PORT", "8080"))
    parser.add_argument(
        "--port", type=int, default=default_port, help="Port to listen on"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        port = args.port
        env_port = os.getenv("PORT")
        if env_port:
            port = int(env_port)

        asyncio.run(_run_app(args.host, port))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        pass


if __name__ == "__main__":
    main()
