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
import pywin32_bootstrap
from engauto_mcp.__main__ import main
raise SystemExit(main())
'@

$bootstrap = $bootstrap.Replace("__ROOT__", $root)

$tmp = Join-Path $env:TEMP "engauto_mcp_server_bootstrap.py"
Set-Content -Path $tmp -Value $bootstrap -Encoding ascii
& $python $tmp
