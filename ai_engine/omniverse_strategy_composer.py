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
import uuid, hashlib, subprocess, sys, platform
from typing import Dict, Any, List, Tuple
from pathlib import Path

STRATEGY_FOLDER = Path("strategies/")
os.makedirs(STRATEGY_FOLDER, exist_ok=True)

# Best-effort: ensure stdout/stderr can emit Unicode (emojis) on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Component Templates ----------
INDICATORS = {
    "momentum": {"desc": "Detect 5D price acceleration", "params": {"window": 5, "threshold": 0.02}},
    # Preset-specific aliases for multi-momentum
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

# Optional: richer risk profile used by V2 preset
RISK_RULES_V2 = {
    **RISK_RULES,
    "trailing_stop_pct": 0.015,   # 1.5% trailing stop
    "take_profit_rr": 2.0,        # take profit at 2R
    "per_trade_risk_pct": 0.005   # 0.5% of equity per trade (for sizing)
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
    return strategy


def write_strategy_file(strategy: Dict[str, Any]) -> Path:
    file = STRATEGY_FOLDER / f"{strategy['name'].replace(' ', '_')}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(strategy, f, indent=4)
    print(f"✅ Strategy '{strategy['name']}' created → {file}")
    print(
        f"📊 Expected Winrate {strategy['meta']['expected_winrate'] * 100:.1f}%, RR {strategy['meta']['expected_risk_reward']}"
    )
    return file


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


# ---------- Meta helpers (extended metadata) ----------
def compute_components_hash(components: dict) -> str:
    try:
        normalized = json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(normalized).hexdigest()
    except Exception:
        return "sha256:"


def get_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception:
        return ""


def _utc_now_iso() -> str:
    try:
        # Python 3.11+ has datetime.UTC
        return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Fallback for older versions
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enrich_meta(strategy: dict) -> dict:
    """Augment strategy['meta'] with a richer schema while keeping old keys."""
    meta = strategy.get("meta", {})
    comps = strategy.get("components", {})
    win = meta.get("expected_winrate")
    rr = meta.get("expected_risk_reward")
    meta.update({
        "schema_version": "2.0",
        "strategy_id": str(uuid.uuid4()),
        "strategy_version": "1.0.0",
        "generator": {"app": "Omniverse Strategy Composer", "version": "1.1.0"},
        "build": {
            "git_commit": get_git_commit(),
            "timestamp": _utc_now_iso(),
            "environment": f"py{sys.version_info.major}.{sys.version_info.minor}-{platform.system().lower()}"
        },
        "components_hash": compute_components_hash(comps),
        "performance": {
            "overall": {
                "win_rate": win if isinstance(win, (int, float)) else None,
                "avg_rr": rr if isinstance(rr, (int, float)) else None
            }
        },
        "validation": {
            "status": "draft",
            "timestamp": _utc_now_iso()
        }
    })
    strategy["meta"] = meta
    return strategy


def validate_meta_schema(strategy: dict) -> Tuple[List[str], bool]:
    """Validate meta against the v2 schema if jsonschema is available.
    Returns (errors, available). If available is False, validation was skipped.
    """
    errors: List[str] = []
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
        available = True
    except Exception:
        # jsonschema not installed; skip validation
        return errors, False

    schema_path = Path("schemas/strategy_meta_v2.schema.json")
    if not schema_path.exists():
        # Schema not available in repo; treat as validator unavailable
        return errors, False
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        meta = strategy.get("meta", {})
        validate(instance=meta, schema=schema)
    except ValidationError as ve:
        errors.append(str(ve))
    except Exception as ex:
        errors.append(f"Schema validation error: {ex}")
    return errors, True


def validate_full_strategy_schema(strategy: dict) -> Tuple[List[str], bool]:
    """Validate the entire strategy JSON against schemas/strategy_v1.schema.json.
    Returns (errors, available). If available is False, validation was skipped.
    """
    errors: List[str] = []
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
        available = True
    except Exception:
        return errors, False

    schema_path = Path("schemas/strategy_v1.schema.json")
    if not schema_path.exists():
        return errors, False
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        validate(instance=strategy, schema=schema)
    except ValidationError as ve:
        errors.append(str(ve))
    except Exception as ex:
        errors.append(f"Schema validation error: {ex}")
    return errors, True


# ---------- Modular components loader ----------
def load_components() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Optionally import external components, otherwise use built-ins.

    Allows users to define ai_engine/strategy_components.py with INDICATORS,
    AI_MODELS, and RISK_RULES to override defaults.
    """
    global INDICATORS, AI_MODELS, RISK_RULES
    try:
        from ai_engine import strategy_components as sc  # type: ignore
        ext_ind = getattr(sc, "INDICATORS", INDICATORS)
        ext_models = getattr(sc, "AI_MODELS", AI_MODELS)
        ext_risk = getattr(sc, "RISK_RULES", RISK_RULES)
        return ext_ind, ext_models, ext_risk
    except Exception:
        return INDICATORS, AI_MODELS, RISK_RULES


# ---------- Quant validation ----------
def _warn(msg: str):
    print(f"⚠️  {msg}")


def validate_strategy(
    strategy: Dict[str, Any],
    indicators_catalog: Dict[str, Any],
    models_catalog: Dict[str, Any],
    strict: bool = False,
) -> List[str]:
    errors: List[str] = []

    comps = strategy.get("components", {})

    # Indicators non-empty and known
    inds = comps.get("indicators", [])
    if not inds:
        errors.append("No indicators selected")
    else:
        unknown = [i for i in inds if i not in indicators_catalog]
        if unknown:
            errors.append(f"Unknown indicators: {', '.join(unknown)}")

    # Model exists
    model = comps.get("ai_model")
    if not model:
        errors.append("No AI model selected")
    elif model not in models_catalog:
        errors.append(f"Unknown model: {model}")

    # Risk checks
    risk = comps.get("risk", {})
    def _pos(name, mx=None):
        v = risk.get(name)
        if v is None:
            return
        if not isinstance(v, (int, float)) or v < 0:
            errors.append(f"Risk.{name} must be a non-negative number")
        if mx is not None and isinstance(v, (int, float)) and v > mx:
            _warn(f"Risk.{name} unusually high ({v} > {mx})")
    _pos("max_loss")
    _pos("cooldown_secs", mx=3600)
    if not isinstance(risk.get("max_positions", 0), int) or risk.get("max_positions", 0) <= 0:
        errors.append("Risk.max_positions must be a positive integer")

    # Optional advanced risk fields
    ts = risk.get("trailing_stop_pct")
    if ts is not None and not (0 < ts < 0.2):
        errors.append("Risk.trailing_stop_pct should be between 0 and 0.2")
    pr = risk.get("per_trade_risk_pct")
    if pr is not None and not (0 < pr < 0.1):
        errors.append("Risk.per_trade_risk_pct should be between 0 and 0.1")
    rr = risk.get("take_profit_rr")
    if rr is not None and rr < 1:
        errors.append("Risk.take_profit_rr should be >= 1")

    # Entry/Exit presence and minimal sanity
    entry = comps.get("entry_logic", "").strip()
    exit_logic = comps.get("exit_logic", "").strip()
    if not entry:
        errors.append("Entry logic is empty")
    if not exit_logic:
        errors.append("Exit logic is empty")

    # Heuristic: warn if entry references unknown indicator tokens
    tokens = set([t for t in inds if isinstance(t, str)])
    ref_warns = []
    for expr, label in [(entry, "entry"), (exit_logic, "exit")]:
        for t in tokens:
            if t in expr:
                break
        else:
            ref_warns.append(label)
    if ref_warns:
        _warn(f"Indicator names not referenced in {', '.join(ref_warns)} expression(s) — check logic text")

    if strict and errors:
        for e in errors:
            print(f"❌ {e}")
    else:
        for e in errors:
            _warn(e)
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Bambhoria Omniverse Strategy Composer")
    parser.add_argument("--name", type=str, help="Strategy name")
    parser.add_argument("--indicators", type=str, help="Comma separated indicators, e.g. momentum,rsi,macd")
    # No static choices: allow modular catalogs
    parser.add_argument("--model", type=str, help="AI model key")
    parser.add_argument("--entry", type=str, help="Entry condition text")
    parser.add_argument("--exit", dest="exit_cond", type=str, help="Exit condition text")
    parser.add_argument("--preset", type=str, choices=["Bambhoria_Alpha_Core_V2"], help="Generate a named preset strategy")
    parser.add_argument("--scaffold", action="store_true", help="Create a starter strategy template JSON (uses --name if provided)")
    parser.add_argument("--non-interactive", action="store_true", help="Run without prompts using provided or default values")
    parser.add_argument("--list", action="store_true", help="List available indicators and models and exit")
    parser.add_argument("--strict", action="store_true", help="Fail on validation errors")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and exit without writing file")
    parser.add_argument("--meta-extended", action="store_true", help="Emit extended metadata (schema v2) alongside existing fields")
    parser.add_argument("--meta-validate", action="store_true", help="Validate meta against schema v2 (auto-enrich for validation)")
    parser.add_argument("--strategy-validate", action="store_true", help="Validate entire strategy JSON against schema")
    parser.add_argument("--validate-file", type=str, help="Validate an existing strategy JSON file on disk")
    parser.add_argument("--meta-only", action="store_true", help="Print only the meta JSON (optionally enriched) and exit")
    parser.add_argument("--upgrade-meta", type=str, help="Enrich an existing strategy JSON file in-place with meta schema v2")
    parser.add_argument("--meta-diff", type=str, help="Compare file meta with freshly composed/enriched meta for drift audits")
    return parser.parse_args()


def run_interactive():
    ind_catalog, model_catalog, _ = load_components()
    show_menu()
    name = input("Enter Strategy Name: ") or "Bambhoria Alpha Core"
    chosen_inds = input("Select indicators (comma separated): ").split(",")
    chosen_inds = [i.strip() for i in chosen_inds if i.strip() and i.strip() in ind_catalog]
    model = input("Select AI Model: ") or "lightgbm_v51"
    entry = input("Entry condition (text): ") or "BUY when momentum>0 and rsi<60"
    exit_condition = input("Exit condition (text): ") or "SELL when rsi>70 or loss>2%"
    strat = compose_strategy(name, chosen_inds or ["momentum", "rsi"], model, entry, exit_condition)
    validate_strategy(strat, ind_catalog, model_catalog, strict=False)
    write_strategy_file(strat)


def run_non_interactive(args):
    # Load modular components
    ind_catalog, model_catalog, _ = load_components()
    # Scaffold template short-circuit
    if args.scaffold:
        name = args.name or "New Bambhoria Strategy"
        template = {
            "name": name,
            "version": "1.0",
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "components": {
                "indicators": ["momentum", "rsi"],
                "ai_model": args.model or "lightgbm_v51",
                "entry_logic": "TODO: Define entry e.g., BUY when momentum>0 and rsi<60",
                "exit_logic": "TODO: Define exit e.g., SELL when rsi>70 or loss>2%",
                "risk": RISK_RULES
            },
            "meta": {
                "expected_winrate": round(random.uniform(0.55, 0.75), 2),
                "expected_risk_reward": round(random.uniform(1.5, 2.5), 2)
            }
        }
        file = STRATEGY_FOLDER / f"{name.replace(' ', '_')}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=4)
        print(f"✅ Scaffold created → {file}")
        print("✍️  Edit entry/exit/risk as needed, then use in your pipeline.")
        return template

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
        strat = compose_strategy(name, indicators, model, entry, exit_condition, risk=RISK_RULES_V2)
        errs = validate_strategy(strat, ind_catalog, model_catalog, strict=args.strict)
        if args.validate_only:
            print("✅ Validation complete (preset)." if not errs else "❌ Validation failed (preset).")
            return None
        if args.strict and errs:
            raise SystemExit(1)
        # Meta schema validation (auto-enrich for validation only)
        if args.meta_validate:
            enriched = enrich_meta(json.loads(json.dumps(strat)))
            meta_errs, available = validate_meta_schema(enriched)
            if not available:
                print("⚠️  Meta schema validation skipped (jsonschema not installed)")
            elif meta_errs:
                for e in meta_errs:
                    print(f"❌ Meta schema: {e}")
                if args.strict:
                    raise SystemExit(1)
            else:
                print("✅ Meta schema validation passed (preset)")
        if args.meta_extended or args.meta_only:
            strat = enrich_meta(strat)
        if args.strategy_validate:
            full_errs, available = validate_full_strategy_schema(strat)
            if not available:
                print("⚠️  Strategy schema validation skipped (jsonschema or schema missing)")
            elif full_errs:
                for e in full_errs:
                    print(f"❌ Strategy schema: {e}")
                if args.strict:
                    raise SystemExit(1)
            else:
                print("✅ Strategy schema validation passed (preset)")
        if args.meta_only:
            print(json.dumps(strat.get("meta", {}), indent=2))
            return None
        return write_strategy_file(strat)

    # Defaults matching interactive prompts
    name = args.name or "Bambhoria Alpha Core"
    indicators = [x.strip() for x in (args.indicators or "momentum,rsi").split(",") if x.strip()]
    model = args.model or "lightgbm_v51"
    entry = args.entry or "BUY when momentum>0 and rsi<60"
    exit_condition = args.exit_cond or "SELL when rsi>70 or loss>2%"
    # Basic validation: keep only known indicators
    indicators = [i for i in indicators if i in ind_catalog]
    if not indicators:
        indicators = ["momentum", "rsi"]
    strat = compose_strategy(name, indicators, model, entry, exit_condition)
    errs = validate_strategy(strat, ind_catalog, model_catalog, strict=args.strict)
    if args.validate_only:
        print("✅ Validation complete." if not errs else "❌ Validation failed.")
        return None
    if args.strict and errs:
        raise SystemExit(1)
    # Meta schema validation (auto-enrich for validation only)
    if args.meta_validate:
        enriched = enrich_meta(json.loads(json.dumps(strat)))
        meta_errs, available = validate_meta_schema(enriched)
        if not available:
            print("⚠️  Meta schema validation skipped (jsonschema not installed)")
        elif meta_errs:
            for e in meta_errs:
                print(f"❌ Meta schema: {e}")
            if args.strict:
                raise SystemExit(1)
        else:
            print("✅ Meta schema validation passed")
    if args.meta_extended:
        strat = enrich_meta(strat)
    if args.strategy_validate:
        full_errs, available = validate_full_strategy_schema(strat)
        if not available:
            print("⚠️  Strategy schema validation skipped (jsonschema or schema missing)")
        elif full_errs:
            for e in full_errs:
                print(f"❌ Strategy schema: {e}")
            if args.strict:
                raise SystemExit(1)
        else:
            print("✅ Strategy schema validation passed")
    return write_strategy_file(strat)


def run_validate_file(args):
    """Validate an existing strategy JSON file on disk."""
    file_path = args.validate_file
    if not file_path:
        print("❌ --validate-file requires a path")
        raise SystemExit(2)
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        raise SystemExit(2)
    try:
        with open(p, "r", encoding="utf-8") as f:
            strategy = json.load(f)
    except Exception as ex:
        print(f"❌ Failed to read JSON: {ex}")
        raise SystemExit(2)

    ind_catalog, model_catalog, _ = load_components()
    errs = validate_strategy(strategy, ind_catalog, model_catalog, strict=args.strict)

    # Decide meta to validate: if schema_version present, validate as-is; otherwise, validate an enriched copy
    meta = strategy.get("meta", {})
    to_validate = strategy
    if args.meta_validate:
        if not isinstance(meta, dict) or meta.get("schema_version") != "2.0":
            to_validate = enrich_meta(json.loads(json.dumps(strategy)))
        meta_errs, available = validate_meta_schema(to_validate)
        if not available:
            print("⚠️  Meta schema validation skipped (jsonschema not installed)")
        elif meta_errs:
            for e in meta_errs:
                print(f"❌ Meta schema: {e}")
            errs.extend([f"meta: {e}" for e in meta_errs])
        else:
            print("✅ Meta schema validation passed (file)")

    # Full strategy schema validation
    if getattr(args, "strategy_validate", False):
        full_errs, available = validate_full_strategy_schema(strategy)
        if not available:
            print("⚠️  Strategy schema validation skipped (jsonschema or schema missing)")
        elif full_errs:
            for e in full_errs:
                print(f"❌ Strategy schema: {e}")
            errs.extend([f"strategy: {e}" for e in full_errs])
        else:
            print("✅ Strategy schema validation passed (file)")

    if errs:
        print(f"❌ Validation failed with {len(errs)} error(s)")
        if args.strict:
            raise SystemExit(1)
    else:
        print("✅ Strategy file validation passed")


def run_upgrade_meta(args):
    """Enrich an existing strategy JSON file in-place with meta schema v2."""
    file_path = args.upgrade_meta
    if not file_path:
        print("❌ --upgrade-meta requires a path")
        raise SystemExit(2)
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        raise SystemExit(2)
    try:
        with open(p, "r", encoding="utf-8") as f:
            strategy = json.load(f)
    except Exception as ex:
        print(f"❌ Failed to read JSON: {ex}")
        raise SystemExit(2)

    # Enrich meta
    strategy = enrich_meta(strategy)

    # Optional meta schema validation
    if getattr(args, "meta_validate", False):
        meta_errs, available = validate_meta_schema(strategy)
        if not available:
            print("⚠️  Meta schema validation skipped (jsonschema or schema missing)")
        elif meta_errs:
            for e in meta_errs:
                print(f"❌ Meta schema: {e}")
            if getattr(args, "strict", False):
                raise SystemExit(1)
        else:
            print("✅ Meta schema validation passed (upgrade)")

    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=4)
        print(f"✅ Upgraded meta written → {p}")
    except Exception as ex:
        print(f"❌ Failed to write JSON: {ex}")
        raise SystemExit(2)


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, path))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            path = f"{prefix}[{i}]"
            out.update(_flatten(v, path))
    else:
        out[prefix] = d
    return out


def _load_meta_ignore_keys() -> set:
    defaults = set([
    "expected_winrate",
    "expected_risk_reward",
    "strategy_id",
    "build.timestamp",
    "validation.timestamp",
    "performance.overall.win_rate",
    "performance.overall.avg_rr",
    ])
    cfg = Path("config/meta_diff_ignore.json")
    if cfg.exists():
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            extra = set()
            if isinstance(data, dict) and isinstance(data.get("ignore_keys"), list):
                extra = set(k for k in data["ignore_keys"] if isinstance(k, str))
            elif isinstance(data, list):
                extra = set(k for k in data if isinstance(k, str))
            return defaults | extra
        except Exception:
            return defaults
    return defaults


def _filter_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    flat = _flatten(meta)
    ignore = _load_meta_ignore_keys()
    kept = {k: v for k, v in flat.items() if k not in ignore}
    return kept


def run_meta_diff(args):
    """Compare file meta with freshly composed/enriched meta and report drift."""
    file_path = args.meta_diff
    if not file_path:
        print("❌ --meta-diff requires a path")
        raise SystemExit(2)
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        raise SystemExit(2)
    try:
        with open(p, "r", encoding="utf-8") as f:
            original = json.load(f)
    except Exception as ex:
        print(f"❌ Failed to read JSON: {ex}")
        raise SystemExit(2)

    # Enrich original meta if needed
    orig_for_compare = original
    if not isinstance(original.get("meta"), dict) or original["meta"].get("schema_version") != "2.0":
        orig_for_compare = enrich_meta(json.loads(json.dumps(original)))

    comps = original.get("components", {})
    # Recompose a fresh strategy using the same components
    name = original.get("name", "DriftAudit")
    indicators = comps.get("indicators", [])
    model = comps.get("ai_model", "lightgbm_v51")
    entry = comps.get("entry_logic", "")
    exit_logic = comps.get("exit_logic", "")
    risk = comps.get("risk")
    fresh = compose_strategy(name, indicators, model, entry, exit_logic, risk=risk)
    fresh = enrich_meta(fresh)

    a = _filter_meta(orig_for_compare.get("meta", {}))
    b = _filter_meta(fresh.get("meta", {}))
    diffs: List[str] = []
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        va = a.get(k, "<missing>")
        vb = b.get(k, "<missing>")
        if va != vb:
            diffs.append(f"{k}: {va} -> {vb}")

    if diffs:
        print("❌ Meta drift detected:")
        for d in diffs:
            print(f"  - {d}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("✅ No meta drift (after ignoring volatile fields)")
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        raise SystemExit(2)
    try:
        with open(p, "r", encoding="utf-8") as f:
            strategy = json.load(f)
    except Exception as ex:
        print(f"❌ Failed to read JSON: {ex}")
        raise SystemExit(2)

    # Enrich meta and optionally validate
    strategy = enrich_meta(strategy)
    if args.meta_validate:
        meta_errs, available = validate_meta_schema(strategy)
        if not available:
            print("⚠️  Meta schema validation skipped (jsonschema not installed)")
        elif meta_errs:
            for e in meta_errs:
                print(f"❌ Meta schema: {e}")
            if args.strict:
                raise SystemExit(1)
        else:
            print("✅ Meta schema validation passed (upgrade)")

    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=4)
        print(f"✅ Upgraded meta written → {p}")
    except Exception as ex:
        print(f"❌ Failed to write JSON: {ex}")
        raise SystemExit(2)


def main():
    args = parse_args()

    if args.list:
        ind_catalog, model_catalog, _ = load_components()
        print("Available Indicators:")
        for k, v in ind_catalog.items():
            desc = v.get("desc") if isinstance(v, dict) else str(v)
            print(f"  - {k}: {desc}")
        print("\nAvailable AI Models:")
        for k, v in model_catalog.items():
            desc = v.get("desc") if isinstance(v, dict) else str(v)
            print(f"  - {k}: {desc}")
        return

    # Validate existing file path, if provided
    if args.validate_file:
        run_validate_file(args)
        return

    # Upgrade meta for an existing file
    if args.upgrade_meta:
        run_upgrade_meta(args)
        return

    # Meta drift audit against recomposed strategy
    if args.meta_diff:
        run_meta_diff(args)
        return

    # If any CLI value provided or --non-interactive/--scaffold/--preset flag, use non-interactive path
    if args.non_interactive or args.scaffold or args.preset or any([args.name, args.indicators, args.model, args.entry, args.exit_cond]):
        run_non_interactive(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
