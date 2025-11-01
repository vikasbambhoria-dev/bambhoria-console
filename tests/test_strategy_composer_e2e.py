import json
import subprocess
import sys
import shutil
from pathlib import Path


def run_cli(args, cwd=None):
    proc = subprocess.run([sys.executable, "ai_engine/omniverse_strategy_composer.py", *args],
                          cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_upgrade_meta_and_meta_diff_e2e(tmp_path: Path):
    src = Path('strategies') / 'Bambhoria_Alpha_Core_V2.json'
    assert src.exists(), "Preset strategy file missing"
    dst = tmp_path / 'Bambhoria_Alpha_Core_V2_copy.json'
    shutil.copy2(src, dst)

    # 1) Upgrade meta in place
    code, out, err = run_cli(["--upgrade-meta", str(dst), "--meta-validate", "--strict"])
    assert code == 0, f"upgrade-meta failed: {out}\n{err}"
    assert "Upgraded meta written" in out

    # 2) Meta diff should report no drift after ignoring volatile fields
    code, out, err = run_cli(["--meta-diff", str(dst), "--strict"])
    assert code == 0, f"meta-diff failed: {out}\n{err}"
    assert "No meta drift" in out

    # 3) Validate file against full strategy schema
    code, out, err = run_cli(["--validate-file", str(dst), "--strategy-validate", "--strict"])
    assert code == 0, f"strategy schema validation failed: {out}\n{err}"
    assert "Strategy file validation passed" in out
