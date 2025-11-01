import json
import shutil
import tempfile
from pathlib import Path

import pytest

from ai_engine import omniverse_strategy_composer as osc


def test_validate_preset_file():
    strat_path = Path('strategies') / 'Bambhoria_Alpha_Core_V2.json'
    assert strat_path.exists(), "Preset strategy file is missing"
    with open(strat_path, 'r', encoding='utf-8') as f:
        strategy = json.load(f)

    ind_catalog, model_catalog, _ = osc.load_components()
    errs = osc.validate_strategy(strategy, ind_catalog, model_catalog, strict=True)
    assert not errs, f"Validation errors: {errs}"


def test_meta_schema_enrichment_validation():
    strat_path = Path('strategies') / 'Bambhoria_Alpha_Core_V2.json'
    with open(strat_path, 'r', encoding='utf-8') as f:
        strategy = json.load(f)

    enriched = osc.enrich_meta(json.loads(json.dumps(strategy)))
    errors, available = osc.validate_meta_schema(enriched)
    # If schema/jsonschema isn't available, validation is skipped gracefully
    if not available:
        pytest.skip("jsonschema or schema file not available; skipping meta schema check")
    assert not errors, f"Meta schema errors: {errors}"


def test_meta_diff_no_drift(tmp_path: Path):
    # Copy the preset to a temp location to avoid modifying the original
    src = Path('strategies') / 'Bambhoria_Alpha_Core_V2.json'
    dst = tmp_path / 'Bambhoria_Alpha_Core_V2_copy.json'
    shutil.copy2(src, dst)

    class Args:
        strict = False
        meta_diff = str(dst)
        meta_validate = False

    # Should not raise and should report no drift after ignoring volatile fields
    osc.run_meta_diff(Args())
    # Ensure file still exists and is valid JSON after potential enrichment write
    with open(dst, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict) and 'meta' in loaded
