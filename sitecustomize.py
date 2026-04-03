from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    resolved = str(path.resolve())
    if path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


ROOT = Path(__file__).resolve().parent
_prepend_path(ROOT / "src")
_prepend_path(ROOT / ".deps")
_prepend_path(ROOT / ".deps" / "win32")
_prepend_path(ROOT / ".deps" / "win32" / "lib")
_prepend_path(ROOT / ".deps" / "pythonwin")
_prepend_path(ROOT / ".deps" / "pywin32_system32")

pywin32_dir = ROOT / ".deps" / "pywin32_system32"
if pywin32_dir.exists():
    os.environ["PATH"] = f"{pywin32_dir};{os.environ.get('PATH', '')}"
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        add_dll_directory(str(pywin32_dir))
