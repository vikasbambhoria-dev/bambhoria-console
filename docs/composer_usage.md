# Omniverse Strategy Composer: Usage Guide

This document explains how to use the strategy composer to generate, validate, and audit strategy blueprints.

## Quick Start

- List available components:

```powershell
python ai_engine/omniverse_strategy_composer.py --list
```

- Generate a preset (Bambhoria Alpha Core V2):

```powershell
python ai_engine/omniverse_strategy_composer.py --non-interactive --preset Bambhoria_Alpha_Core_V2
```

- Validate-only (no file write):

```powershell
python ai_engine/omniverse_strategy_composer.py --non-interactive --preset Bambhoria_Alpha_Core_V2 --validate-only
```

## Metadata (Schema v2)

- Emit extended metadata (adds schema_version, strategy_id, build info, components_hash, and nested performance):

```powershell
python ai_engine/omniverse_strategy_composer.py --non-interactive --preset Bambhoria_Alpha_Core_V2 --meta-extended
```

- Validate metadata against JSON Schema:

```powershell
python ai_engine/omniverse_strategy_composer.py --non-interactive --preset Bambhoria_Alpha_Core_V2 --meta-validate
```

Notes:
- `--meta-validate` auto-enriches in-memory if meta v2 is not present, so you can validate without writing.
- Use `--strict` to fail on validation errors.

## Validate an Existing Strategy File

- Validate a JSON already on disk (components + optional meta schema):

```powershell
python ai_engine/omniverse_strategy_composer.py --validate-file strategies\Bambhoria_Alpha_Core_V2.json --meta-validate --strict
```

- Validation behavior:
  - Components are checked for known indicators/models and sane risk fields.
  - If meta.schema_version == "2.0", the meta block is validated as-is; otherwise, a copy is auto-enriched and validated.
  - If `jsonschema` is not installed, meta validation is skipped with a warning.

## Flags Summary

- `--non-interactive` Run without prompts using provided/default values
- `--preset` Generate a named preset strategy
- `--name, --indicators, --model, --entry, --exit` Manual composition inputs
- `--list` Print available indicators and models
- `--validate-only` Validate and exit without writing a file
- `--strict` Exit non-zero on validation errors
- `--meta-extended` Write extended metadata fields (schema v2)
- `--meta-validate` Validate meta against `schemas/strategy_meta_v2.schema.json`
- `--validate-file` Validate an existing strategy file on disk
 - `--meta-only` Print only the meta JSON (optionally enriched) and exit
 - `--upgrade-meta` Enrich an existing strategy JSON file in-place with meta schema v2
 - `--meta-diff` Compare file meta against a freshly composed/enriched meta for drift audits (honors `--strict`)

## Schema

- JSON Schema file: `schemas/strategy_meta_v2.schema.json`

## Tips

- The extended meta includes `components_hash` which uniquely fingerprints the components block; any changes will produce a different hash.
- Use `--meta-validate --strict` in CI to ensure strategies shipped to prod meet the metadata contract.

## CI: Validate strategies on PR/push

This repo includes a workflow that validates all strategy files automatically:

- File: `.github/workflows/validate_strategies.yml`
- Behavior:
  - Runs on push and pull_request
  - Installs `jsonschema`
  - Validates each `strategies/*.json` with:
    - `--validate-file <path> --meta-validate --strict`
  - Fails the check if any file is invalid

Local dry-run equivalent:

```powershell
# Validate a single file strictly
python ai_engine/omniverse_strategy_composer.py --validate-file strategies\\Bambhoria_Alpha_Core_V2.json --meta-validate --strict
```

## Pre-commit Hook (optional)

Install a git pre-commit hook to validate staged strategy files:

```powershell
# From repo root
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_precommit_hook.ps1
```

This installs a shim at `.git/hooks/pre-commit` that runs `scripts/hooks/pre-commit.ps1` to validate staged `strategies/*.json` files with `--meta-validate --strict`.

Notes:
- If the folder is not a git repository yet, the installer now exits gracefully and prints an initialization tip. You can run:

```powershell
git init; git add .; git commit -m "init"; pwsh -NoProfile -File scripts/install_precommit_hook.ps1
```

## Meta Drift Audit

Compare a file’s meta with a freshly composed/enriched meta (ignores volatile fields like timestamps and expected winrate):

```powershell
python ai_engine/omniverse_strategy_composer.py --meta-diff strategies\Bambhoria_Alpha_Core_V2.json --strict
```

## Developer tests

Run a small, fast test suite for the composer and meta utilities:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_strategy_composer_meta.py
```
