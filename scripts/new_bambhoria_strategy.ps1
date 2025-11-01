<#
new_bambhoria_strategy.ps1
Create or update `ai_engine/omniverse_strategy_composer.py` scaffold template.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\new_bambhoria_strategy.ps1 [-Force]

This script will create `ai_engine/omniverse_strategy_composer.py` if it doesn't exist.
If it exists, run with -Force to overwrite.
#>

param(
    [switch]$Force
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetRel = "ai_engine/omniverse_strategy_composer.py"
$target = Join-Path -Path $scriptDir -ChildPath "..\$targetRel" | Resolve-Path -Relative | ForEach-Object { (Resolve-Path $_).ProviderPath }

# If Resolve-Path above fails, compute directly
if (-not $target) {
    $target = Join-Path -Path $scriptDir -ChildPath "..\$targetRel"
}

$target = [System.IO.Path]::GetFullPath($target)

Write-Host "Target file:" $target

if (Test-Path $target -PathType Leaf) {
    if (-not $Force) {
        Write-Host "File already exists. Use -Force to overwrite." -ForegroundColor Yellow
        exit 0
    }
    else {
        Write-Host "Overwriting existing file (force)" -ForegroundColor Cyan
    }
}
else {
    # Ensure directory exists
    $dir = Split-Path -Parent $target
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Write-Host "Creating scaffold file..." -ForegroundColor Green
}

$tmpl = @'
"""
omniverse_strategy_composer.py
Bambhoria Omniverse Strategy Composer v1.0
Author: Vikas Bambhoria
Purpose:
 - Unified Strategy Builder for Bambhoria God-Eye OS
 - Combine Indicators + AI Models + Risk Rules visually
 - Generate executable JSON Strategy Blueprints
"""

import json, os, random, datetime, argparse
from pathlib import Path

STRATEGY_FOLDER = Path("strategies/")
os.makedirs(STRATEGY_FOLDER, exist_ok=True)

# ---------- Component Templates ----------
INDICATORS = {
    "momentum": {"desc": "Detect 5D price acceleration", "params": {"window": 5, "threshold": 0.02}},
    "momentum_fast5": {"desc": "Fast momentum (5 bars)", "params": {"window": 5, "threshold": 0.02}},
    "momentum_slow20": {"desc": "Slow momentum (20 bars)", "params": {"window": 20, "threshold": 0.01}},
    "rsi": {"desc": "Relative Strength Index", "params": {"period": 14, "upper": 70, "lower": 30}},
    "macd": {"desc": "Moving Average Convergence Divergence", "params": {"fast": 12, "slow": 26, "signal": 9}},
    "volume_surge": {"desc": "Detect sudden volume rise", "params": {"multiplier": 1.5}},
}

AI_MODELS = {
    "lightgbm_v51": {"desc": "Core price-action predictor"},
    "neural_insight_v1": {"desc": "Pattern-optimized model from trade logs"},
    "reinforce_qnet": {"desc": "Reinforcement-learning Q-network (prototype)"}
}

RISK_RULES = {
    "max_loss": 5000,
    "max_positions": 100,
    "cooldown_secs": 10
}

RISK_RULES_V2 = {
    **RISK_RULES,
    "trailing_stop_pct": 0.015,
    "take_profit_rr": 2.0,
    "per_trade_risk_pct": 0.005
}

# ---------- Strategy Builder ----------
def compose_strategy(name: str, indicators: list, model: str, entry: str, exit: str, risk=None):
    strategy = {
        "name": name,
        "version": "1.0",
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "components": {
            "indicators": indicators,
            "ai_model": model,
            "entry_logic": entry,
            "exit_logic": exit,
            "risk": risk or RISK_RULES
        },
        "meta": {
            "expected_winrate": round(random.uniform(0.55, 0.85), 2),
            "expected_risk_reward": round(random.uniform(1.5, 3.0), 2)
        }
    }
    file = STRATEGY_FOLDER / f"{name.replace(' ', '_')}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(strategy, f, indent=4)
    print(f"✅ Strategy '{name}' created → {file}")
    print(f"📊 Expected Winrate {strategy['meta']['expected_winrate'] * 100:.1f}%, RR {strategy['meta']['expected_risk_reward']}")
    return strategy


# ---------- Visual Composer (CLI simulation) ----------
def show_menu():
    print("\n🪶 Bambhoria Omniverse Strategy Composer v1.0")
    print("------------------------------------------------")
    print("Available Indicators:")
    for key, value in INDICATORS.items():
        print(f"  - {key}: {value['desc']}")
    print("\nAvailable AI Models:")
    for key, value in AI_MODELS.items():
        print(f"  - {key}: {value['desc']}")
    print("\nRisk Defaults:", RISK_RULES)
    print("------------------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description="Bambhoria Omniverse Strategy Composer")
    parser.add_argument("--name", type=str, help="Strategy name")
    parser.add_argument("--indicators", type=str, help="Comma separated indicators, e.g. momentum,rsi,macd")
    parser.add_argument("--model", type=str, choices=list(AI_MODELS.keys()), help="AI model key")
    parser.add_argument("--entry", type=str, help="Entry condition text")
    parser.add_argument("--exit", dest="exit_cond", type=str, help="Exit condition text")
    parser.add_argument("--preset", type=str, choices=["Bambhoria_Alpha_Core_V2"], help="Generate a named preset strategy")
    parser.add_argument("--non-interactive", action="store_true", help="Run without prompts using provided or default values")
    return parser.parse_args()


def run_interactive():
    show_menu()
    name = input("Enter Strategy Name: ") or "Bambhoria Alpha Core"
    chosen_inds = input("Select indicators (comma separated): ").split(",")
    chosen_inds = [i.strip() for i in chosen_inds if i.strip()]
    model = input("Select AI Model: ") or "lightgbm_v51"
    entry = input("Entry condition (text): ") or "BUY when momentum>0 and rsi<60"
    exit_condition = input("Exit condition (text): ") or "SELL when rsi>70 or loss>2%"
    compose_strategy(name, chosen_inds, model, entry, exit_condition)


def run_non_interactive(args):
    # Preset path overrides free-form inputs
    if args.preset == "Bambhoria_Alpha_Core_V2":
        name = "Bambhoria Alpha Core V2"
        indicators = [
            "momentum_fast5",
            "momentum_slow20",
            "rsi",
            "volume_surge"
        ]
        model = args.model or "lightgbm_v51"
        entry = (
            "BUY when momentum_fast5>0 AND momentum_slow20>0 "
            "AND rsi<60 AND volume_surge==true"
        )
        exit_condition = (
            "SELL when rsi>70 OR loss>1.5% OR (momentum_fast5<0 AND momentum_slow20<0 for 3 bars) "
            "OR trailing_stop_hit"
        )
        return compose_strategy(name, indicators, model, entry, exit_condition, risk=RISK_RULES_V2)

    # Defaults matching interactive prompts
    name = args.name or "Bambhoria Alpha Core"
    indicators = [x.strip() for x in (args.indicators or "momentum,rsi").split(",") if x.strip()]
    model = args.model or "lightgbm_v51"
    entry = args.entry or "BUY when momentum>0 and rsi<60"
    exit_condition = args.exit_cond or "SELL when rsi>70 or loss>2%"
    # Basic validation: keep only known indicators
    indicators = [i for i in indicators if i in INDICATORS]
    if not indicators:
        indicators = ["momentum", "rsi"]
    compose_strategy(name, indicators, model, entry, exit_condition)


def main():
    args = parse_args()
    # If any CLI value provided or --non-interactive flag, use non-interactive path
    if args.non_interactive or any([args.name, args.indicators, args.model, args.entry, args.exit_cond]):
        run_non_interactive(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
'@

Set-Content -Path $target -Value $tmpl -Encoding UTF8

Write-Host "Done. File written to $target" -ForegroundColor Green
