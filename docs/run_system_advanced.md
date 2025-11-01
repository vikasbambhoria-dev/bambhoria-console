# Run System (Advanced)

This guide launches the full Bambhoria trading ecosystem: dashboards, mock feed, trading pipeline, and quantum monitor, with health checks and live monitoring.

## Prerequisites

- Windows PowerShell 5.1+ (or PowerShell 7)
- Python 3.11+ (venv recommended)
- Dependencies installed:
  ```powershell
  python -m pip install -r requirements.txt
  ```

## One-liner

```powershell
# From repo root
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_system_advanced.ps1
```

- This activates `.venv` if present, sets UTF-8 output, ensures a `logs/` folder, and runs the launcher (`complete_system_launcher_v2.py`).

Headless/server-friendly flags:

```powershell
# Don't open dashboards in the browser
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_system_advanced.ps1 -NoOpen

# Disable watchdog auto-restart (not recommended for prod)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_system_advanced.ps1 -NoWatchdog

# Combine
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_system_advanced.ps1 -NoOpen -NoWatchdog
```

## What starts

- System Monitor Dashboard (http://localhost:5008)
- Analytics Dashboard (http://localhost:5006)
- Performance Monitor (http://localhost:5009)
- Mock Feed Server (http://localhost:8080)
- Complete Trading System (pipeline)
- Quantum Performance Monitor (background)

Logs for each component stream to `logs/*.log`.

## Ports and endpoints

- Configure ports and the mock feed API endpoint without changing code via `config/ports.json`:

```json
{
  "system_monitor": 5008,
  "analytics_dashboard": 5000,
  "performance_monitor": 5009,
  "mock_feed_api": "http://localhost:5002/api/ticks"
}
```

- Environment overrides (take precedence):
  - `BAMBHORIA_PORT_SYSTEM_MONITOR`
  - `BAMBHORIA_PORT_ANALYTICS`
  - `BAMBHORIA_PORT_PERF`
  - `BAMBHORIA_MOCK_FEED_API`

Example (PowerShell):

```powershell
$env:BAMBHORIA_PORT_ANALYTICS=7000
$env:BAMBHORIA_MOCK_FEED_API='http://localhost:6001/api/ticks'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_system_advanced.ps1
```

## Stop

- Press Ctrl+C in the PowerShell window. The launcher gracefully terminates all services.

## Troubleshooting

- Port in use warnings: The launcher will continue but note which port is busy. You can stop any conflicting process or change ports in `config/ports.json` (or env overrides).
  - Prefer configuring ports in `config/ports.json` or via environment variables described above.
- Emoji/Unicode issues: The script sets `PYTHONIOENCODING` to `utf-8`. If your console still cannot render, ignore the icons; logs will still contain text.
- Missing venv: Create and install dependencies:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  ```

## Optional: Git hooks and validation

- Validate strategy JSON files and metadata before committing:
  ```powershell
  pwsh -NoProfile -File scripts/install_precommit_hook.ps1
  ```

- Run focused composer tests:
  ```powershell
  python -m pip install -r requirements-dev.txt
  python -m pytest -q tests/test_strategy_composer_meta.py tests/test_strategy_composer_e2e.py
  ```
