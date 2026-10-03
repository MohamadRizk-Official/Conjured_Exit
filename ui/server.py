"""FastAPI backend for the Pathcaster operator UI.

Run (from the project root, native Windows, x64 venv)::

    cf64\\Scripts\\python.exe -m ui.server --sim          # hardware-free simulator
    cf64\\Scripts\\python.exe -m ui.server --live         # real drone link (config.LINK_URI), connects at startup
    cf64\\Scripts\\python.exe -m ui.server --sim --live   # starts in SIM; the SIM/LIVE toggle switches
    cf64\\Scripts\\python.exe -m ui.server                # neither: tracker/flight publish on ui.state.BUS

then open http://127.0.0.1:8765/ .  Routes:

    GET  /                      the built frontend (ui/web/dist/index.html)
    GET  /api/state             AppState snapshot (JSON)
    GET  /api/config            geofence box, link URI, modes, command vocabulary, available sources
    GET  /api/source            {source, available, connecting, live_connected}
    GET  /api/paths             [{name, mode, length_m, duration_s, n_points}, ...]
    GET  /api/paths/{name}      full path: points [[x,y,z]...], times [...], meta
    POST /api/command           {"name": "...", "args": {...}}  -> queued for the flight loop
                                ("set_source" is handled here by the SourceManager, not queued)
    WS   /ws                    {"type":"state","state":{...}} at <= 20 Hz when the state
                                changes (full snapshot each time, keepalive every 2 s);
                                accepts {"type":"command","name":..,"args":..} and
                                {"type":"ping"}.

Sources: exactly one producer feeds drone/link/flight on the bus at a time -- the
simulator (``ui.sim``), the live bridge (``ui.live``, real cflib link) or nobody.
``SourceManager`` switches between them; a LIVE connect runs in a worker thread so
the UI stays responsive, and falls back to the previous source if it fails.

Wiring for the real system: ``create_app()`` uses ``ui.state.BUS`` / ``COMMANDS``
by default, so tracker/feed/flight/voice just import those two objects, run in
the same process as this server (``threading.Thread(target=run_server)``) or
start the server from their main, and the UI shows whatever they publish.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from ui.pathstore import PathStore
from ui.state import BUS, COMMAND_SPECS, COMMANDS, MODES, SOURCES, CommandQueue, StateBus

__all__ = ["SourceManager", "create_app", "run_server", "main", "WS_MAX_HZ", "WS_KEEPALIVE_S"]

HERE = pathlib.Path(__file__).resolve().parent
DIST_DIR = HERE / "web" / "dist"
WS_MAX_HZ = 20.0
WS_KEEPALIVE_S = 2.0

_FALLBACK_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Pathcaster UI</title>
<style>body{background:#0b0e14;color:#e6e9ef;font:20px system-ui;padding:3rem;max-width:50rem}code{color:#9cf}</style>
</head><body><h1>Pathcaster UI backend is running</h1>
<p>The frontend has not been built yet. From the project root run:</p>
<pre><code>cd ui\\web
npm install
npm run build</code></pre>
<p>then reload this page. The API is live meanwhile: <a href="/api/state" style="color:#9cf">/api/state</a>,
<a href="/api/paths" style="color:#9cf">/api/paths</a>, <a href="/api/config" style="color:#9cf">/api/config</a>.</p>
</body></html>"""

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0b0e14"/>'
    '<polyline points="5,24 12,10 19,20 27,7" fill="none" stroke="#4dd2ff" stroke-width="3" stroke-linecap="round"/>'
    '<circle cx="27" cy="7" r="3.5" fill="#ff4d4d"/></svg>'
)


class CommandIn(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    source: str = "ui"


# ------------------------------------------------------------- source switch


class SourceManager:
    """Decides who feeds the bus: the simulator (``sim``), the live bridge (``live``) or nobody.

    ``sim`` needs ``pause()`` / ``resume()`` / ``stop()``; ``live`` needs ``connected``,
    ``connect()`` (blocking, raises), ``start()``, ``pause()``, ``close()`` -- i.e. ``ui.sim.Simulator``
    and ``ui.live.LiveBridge`` (or fakes in tests).
    """

    def __init__(self, bus: StateBus, commands: CommandQueue, sim: Any = None, live: Any = None) -> None:
        self.bus = bus
        self.commands = commands
        self.sim = sim
        self.live = live
        self.source = "none"
        self._lock = threading.RLock()
        self._connecting = False
        self._worker: threading.Thread | None = None
        bus.update(source="none")

    @property
    def available(self) -> list[str]:
        out = []
        if self.sim is not None:
            out.append("sim")
        if self.live is not None:
            out.append("live")
        out.append("none")
        return out

    @property
    def connecting(self) -> bool:
        return self._connecting

    def status(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "connecting": self._connecting,
            "live_connected": bool(self.live is not None and getattr(self.live, "connected", False)),
        }

    def set_source(self, source: str) -> dict[str, Any]:
        source = str(source or "").strip().lower()
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        with self._lock:
            if self._connecting:
                raise ValueError("LIVE connect in progress; wait for it to finish")
            if source == self.source:
                return self.status()
            if source == "sim":
                self._to_sim()
            elif source == "live":
                self._to_live()
            else:
                self._to_none()
            return self.status()

    # -- transitions (lock held) ------------------------------------------

    def _set(self, source: str) -> None:
        self.source = source
        self.bus.update(source=source)

    def _reset_shared_state(self) -> None:
        """Clear what the previous producer left behind so stale fake values never look real."""
        self.bus.update(
            replay={"active": False, "t": 0.0, "duration": 0.0, "progress": 0.0},
            alarm={"active": False, "blocked_exits": []},
            recording={"active": False},
            wand={"ok": False},
            hwcheck={"active": False},
            tracking={"ok": False, "fps": 0.0, "latency_ms": 0.0},
            flight={"state": "idle", "estimator_converged": False},
            link={"connected": False, "connecting": False, "battery_v": 0.0, "rssi": 0, "error": ""},
        )

    def _to_sim(self) -> None:
        if self.sim is None:
            raise ValueError("no simulator in this process; start the server with --sim")
        if self.live is not None:
            self.live.pause()
        self._reset_shared_state()
        self._set("sim")
        self.sim.resume()
        self.bus.log("source -> SIM (fake drone)")

    def _to_none(self) -> None:
        if self.live is not None:
            self.live.pause()
        if self.sim is not None:
            self.sim.pause()
        self._reset_shared_state()
        self._set("none")
        self.bus.log("source -> none (external tracker/flight publish directly)")

    def _to_live(self) -> None:
        if self.live is None:
            raise ValueError("no live bridge in this process; start the server with --live")
        fallback = self.source
        if self.sim is not None:
            self.sim.pause()
        self._reset_shared_state()
        self._set("live")
        if getattr(self.live, "connected", False):
            self.live.start()
            self.bus.log("source -> LIVE (real drone link)")
            return
        self._connecting = True
        self.bus.update(link={"connecting": True, "uri": getattr(self.live, "uri", config.LINK_URI)})
        self._worker = threading.Thread(target=self._connect_worker, args=(fallback,), name="live-connect", daemon=True)
        self._worker.start()

    def _connect_worker(self, fallback: str) -> None:
        try:
            self.live.connect()
            self.live.start()
            self.bus.log("source -> LIVE (real drone link)")
            with self._lock:
                self._connecting = False
        except Exception as exc:  # noqa: BLE001 - reported on the bus, UI falls back
            err = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._connecting = False
                if fallback == "sim" and self.sim is not None:
                    self._set("sim")
                    self.sim.resume()
                else:
                    fallback = "none"
                    self._set("none")
                # stays visible next to the SIM/LIVE toggle until the next source switch
                self.bus.update(link={"connecting": False, "connected": False, "error": f"LIVE unavailable: {err}"})
            self.bus.log(f"LIVE unavailable ({err}); back to {fallback.upper()}")

    def close(self) -> None:
        with self._lock:
            if self.live is not None:
                try:
                    self.live.close()
                except Exception:  # noqa: BLE001
                    pass
            if self.sim is not None:
                try:
                    self.sim.stop()
                except Exception:  # noqa: BLE001
                    pass


# ----------------------------------------------------------------------- app


def create_app(
    bus: StateBus | None = None,
    commands: CommandQueue | None = None,
    store: PathStore | None = None,
    dist_dir: pathlib.Path | str | None = None,
    manager: SourceManager | None = None,
) -> FastAPI:
    """Build the FastAPI app.  Defaults wire the process-wide ``BUS``/``COMMANDS``."""
    bus = bus if bus is not None else BUS
    commands = commands if commands is not None else COMMANDS
    store = store if store is not None else PathStore(config.PATHS_DIR, demo=False)
    manager = manager if manager is not None else SourceManager(bus, commands)
    dist = pathlib.Path(dist_dir) if dist_dir is not None else DIST_DIR

    app = FastAPI(title="Pathcaster UI", version="0.2.0", docs_url="/api/docs", redoc_url=None)
    app.state.bus = bus
    app.state.commands = commands
    app.state.store = store
    app.state.manager = manager

    # the path list is part of the streamed state; publish it once at startup
    bus.update(paths=store.summaries())
    if not bus.snapshot().link.uri:
        bus.update(link={"uri": config.LINK_URI})

    def apply_ui_side_effects(name: str, args: dict[str, Any]) -> None:
        """Commands that are pure UI state are applied immediately (and still queued)."""
        if name == "select_path":
            p = store.get(str(args.get("name", "")))
            if p is not None:
                bus.update(active_path=p.name)
        elif name == "set_mode":
            mode = str(args.get("mode", "")).lower()
            if mode in MODES:
                bus.update(mode=mode)

    def enqueue(name: str, args: dict[str, Any], source: str) -> dict[str, Any]:
        if name == "set_source":  # infrastructure command: handled here, never queued
            status = manager.set_source(str(args.get("source", "")))  # ValueError on bad/unavailable
            bus.log(f"[{source}] set_source {status['source']}") if source != "ui" else None
            cmd = {"name": name, "args": dict(args), "ts": time.time(), "source": source}
            return {"ok": True, "command": cmd, "queued": len(commands), "status": status}
        cmd = commands.push(name, args, source=source)  # ValueError on bad name/args
        apply_ui_side_effects(cmd.name, cmd.args)
        return {"ok": True, "command": cmd.to_json(), "queued": len(commands)}

    # ------------------------------------------------------------------ pages

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        index_file = dist / "index.html"
        if index_file.is_file():
            return FileResponse(index_file, media_type="text/html")
        return HTMLResponse(_FALLBACK_HTML)

    assets = dist / "assets"
    app.mount("/assets", StaticFiles(directory=str(assets), check_dir=False), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        return Response(_FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"})

    # -------------------------------------------------------------------- api

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": bus.version, "paths": len(store.names()), "source": manager.source}

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(bus.snapshot_dict())

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return {
            "geofence": store.geofence(),
            "link_uri": config.LINK_URI,
            "modes": list(MODES),
            "sources": manager.available,
            "commands": {k: {"args": list(a), "help": h} for k, (a, h) in COMMAND_SPECS.items()},
            "ws_max_hz": WS_MAX_HZ,
            "demo": store.demo,
            "paths_dir": str(store.directory),
        }

    @app.get("/api/source")
    async def get_source() -> dict[str, Any]:
        return manager.status()

    @app.get("/api/paths")
    async def get_paths() -> list[dict[str, Any]]:
        return store.summaries()

    @app.get("/api/paths/{name}")
    async def get_path(name: str) -> JSONResponse:
        p = store.get(name)
        if p is None:
            raise HTTPException(404, f"no path {name!r}; have {store.names()}")
        return JSONResponse(p.to_dict())

    @app.post("/api/paths/refresh")
    async def refresh_paths() -> list[dict[str, Any]]:
        store.refresh()
        summaries = store.summaries()
        bus.update(paths=summaries)
        return summaries

    @app.post("/api/command")
    async def post_command(cmd: CommandIn) -> dict[str, Any]:
        try:
            return enqueue(cmd.name, cmd.args, cmd.source or "ui")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # --------------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        sub = bus.subscribe()  # bound to this event loop
        min_period = 1.0 / WS_MAX_HZ

        async def sender() -> None:
            while True:
                await sub.wait_async(timeout=WS_KEEPALIVE_S)
                snap = sub.take()  # keepalive resends the unchanged snapshot
                await ws.send_text(json.dumps({"type": "state", "state": snap}, separators=(",", ":")))
                await asyncio.sleep(min_period)

        async def receiver() -> None:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad json"}))
                    continue
                kind = msg.get("type", "command")
                if kind == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": msg.get("ts")}))
                    continue
                if kind != "command":
                    await ws.send_text(json.dumps({"type": "error", "error": f"unknown message type {kind!r}"}))
                    continue
                try:
                    result = enqueue(str(msg.get("name", "")), dict(msg.get("args") or {}), str(msg.get("source") or "ui"))
                    await ws.send_text(json.dumps({"type": "ack", **result}))
                except (ValueError, TypeError) as exc:
                    await ws.send_text(json.dumps({"type": "error", "error": str(exc)}))

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc is not None and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                    print(f"[ws] {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            for t in tasks:
                t.cancel()
            sub.close()

    return app


# ------------------------------------------------------------------ runners


def run_server(app: FastAPI, host: str = "127.0.0.1", port: int = 8765, log_level: str = "info") -> None:
    """Blocking uvicorn run (call from a thread if the flight loop owns the main thread)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=log_level)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ui.server", description="Pathcaster operator UI backend")
    ap.add_argument("--sim", action="store_true", help="run the hardware-free simulator (demo paths, fake drone)")
    ap.add_argument("--live", action="store_true", help="real drone link via flight.connect(config.LINK_URI)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the UI from a phone on the same Wi-Fi")
    ap.add_argument("--paths-dir", default=config.PATHS_DIR)
    ap.add_argument("--uri", default=None, help=f"--live only: override config.LINK_URI ({config.LINK_URI})")
    ap.add_argument("--time-scale", type=float, default=1.0, help="--sim only: speed up the fake flight")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    store = PathStore(args.paths_dir, demo=args.sim)
    sim = None
    live = None
    if args.sim:
        from ui.sim import Simulator

        sim = Simulator(BUS, COMMANDS, store, time_scale=args.time_scale)
    if args.live:
        from ui.live import LiveBridge

        live = LiveBridge(BUS, COMMANDS, uri=args.uri)
    manager = SourceManager(BUS, COMMANDS, sim=sim, live=live)
    app = create_app(BUS, COMMANDS, store, manager=manager)

    if args.sim:
        manager.set_source("sim")  # with --sim --live the server starts in SIM; the toggle switches
        print(f"[ui] simulator running with paths: {', '.join(store.names())}")
    elif args.live:
        manager.set_source("live")  # connects in a worker thread; the page shows "connecting..."
        print(f"[ui] LIVE: connecting to {live.uri} in the background")

    print(f"[ui] open http://{args.host}:{args.port}/   (frontend built: {(DIST_DIR / 'index.html').is_file()})")
    try:
        run_server(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
