"""Thread-safe view of the stored paths for the UI server and the simulator.

Saved paths come from ``paths/*.json`` via ``paths.load_path``; with
``demo=True`` the synthetic ``paths.demo_paths()`` are added underneath them (a
saved path with the same name wins).  Names are slugs (``"Exit A" -> "exit_a"``).
"""

from __future__ import annotations

import pathlib
import threading
from typing import Any

import config
import paths as pathlib_engine
from paths import GEOFENCE, Path, demo_paths, list_paths, load_path, save_path, slugify

__all__ = ["PathStore", "summarize"]


def summarize(path: Path) -> dict[str, Any]:
    """The ``AppState.paths`` entry for one path (plain types only)."""
    return {
        "name": str(path.name),
        "mode": str(path.mode),
        "length_m": round(float(path.length), 4),
        "duration_s": round(float(path.duration), 3),
        "n_points": int(len(path.points)),
    }


class PathStore:
    def __init__(self, directory: str | pathlib.Path = config.PATHS_DIR, demo: bool = False) -> None:
        self.directory = pathlib.Path(directory)
        self.demo = bool(demo)
        self._lock = threading.Lock()
        self._paths: dict[str, Path] = {}
        self.refresh()

    # ------------------------------------------------------------------ reads

    def refresh(self) -> dict[str, Path]:
        """Re-read the directory (and the demo set) and return name -> Path."""
        found: dict[str, Path] = {}
        if self.demo:
            for name, p in demo_paths().items():
                found[slugify(name)] = p
        for name in list_paths(self.directory):
            try:
                p = load_path(name, self.directory)
            except (OSError, ValueError) as exc:  # a corrupt file must not kill the UI
                print(f"[pathstore] skipping {name}: {exc}")
                continue
            p.name = slugify(p.name or name)
            found[slugify(name)] = p
        with self._lock:
            self._paths = found
            return dict(found)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._paths)

    def get(self, name: str) -> Path | None:
        with self._lock:
            return self._paths.get(slugify(name))

    def all(self) -> dict[str, Path]:
        with self._lock:
            return dict(self._paths)

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [summarize(self._paths[k]) for k in sorted(self._paths)]

    def exit_path(self, exit_letter: str) -> Path | None:
        """``"A" -> exit_a``, ``"B" -> exit_b`` (whatever is stored under that name)."""
        return self.get(f"exit_{exit_letter.strip().lower()}")

    # ----------------------------------------------------------------- writes

    def save(self, path: Path) -> str:
        """Persist ``path`` under ``paths/<slug>.json`` and refresh. Returns the file."""
        path.name = slugify(path.name)
        file = save_path(path, self.directory)
        self.refresh()
        return file

    @staticmethod
    def geofence() -> dict[str, float]:
        return GEOFENCE.to_dict()

    @staticmethod
    def engine() -> Any:
        """The ``paths`` module (so callers can reach clean_path etc. without re-importing)."""
        return pathlib_engine
