$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Program Files\pgAdmin 4\python\python.exe"
$bootstrap = @'
import os
import sys

ROOT = r'''__ROOT__'''
sys.path[:0] = [
    ROOT + r"\.deps",
    ROOT + r"\.deps\win32",
    ROOT + r"\.deps\win32\lib",
    ROOT + r"\.deps\pythonwin",
    ROOT + r"\src",
]
os.environ["PATH"] = ROOT + r"\.deps\pywin32_system32;" + os.environ["PATH"]
os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
import pywin32_bootstrap
from streamlit.web.cli import main as streamlit_main

sys.argv = [
    "streamlit",
    "run",
    ROOT + r"\src\engauto_mcp\dashboard.py",
    "--global.developmentMode=false",
    "--server.port=8501",
]
raise SystemExit(streamlit_main())
'@

$bootstrap = $bootstrap.Replace("__ROOT__", $root)

$tmp = Join-Path $env:TEMP "engauto_mcp_dashboard_bootstrap.py"
Set-Content -Path $tmp -Value $bootstrap -Encoding ascii
& $python $tmp
