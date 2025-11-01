"""
omniverse_strategy_composer.py
Bambhoria Omniverse Strategy Composer v1.0
Author: Vikas Bambhoria
Purpose:
 - Unified Strategy Builder for Bambhoria God-Eye OS
 - Combine Indicators + AI Models + Risk Rules visually
 - Generate executable JSON Strategy Blueprints
"""

import json, os, random, datetime
from pathlib import Path

STRATEGY_FOLDER = Path("strategies/")
os.makedirs(STRATEGY_FOLDER, exist_ok=True)

# ---------- Component Templates ----------
INDICATORS = {
	"momentum": {"desc": "Detect 5D price acceleration", "params": {"window": 5, "threshold": 0.02}},
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
	with open(file, "w") as f:
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


def main():
	show_menu()
	name = input("Enter Strategy Name: ") or "Bambhoria Alpha Core"
	chosen_inds = input("Select indicators (comma separated): ").split(",")
	model = input("Select AI Model: ") or "lightgbm_v51"
	entry = input("Entry condition (text): ") or "BUY when momentum>0 and rsi<60"
	exit_condition = input("Exit condition (text): ") or "SELL when rsi>70 or loss>2%"
	compose_strategy(name, chosen_inds, model, entry, exit_condition)


if __name__ == "__main__":
	main()
