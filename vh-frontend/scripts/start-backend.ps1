$ErrorActionPreference = "Stop"
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
$backendRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\vh-backend")

uv run --directory $backendRoot uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
exit $LASTEXITCODE
