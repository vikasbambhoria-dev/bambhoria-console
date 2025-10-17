#!/usr/bin/env bash
set -euo pipefail

REPO=""
PRIVATE=0
CREATE_TEST_PR=0

usage() {
  cat <<E
Usage: $0 --repo owner/repo [--private] [--create-test-pr]
E
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) shift; REPO="$1"; shift;;
    --repo=*) REPO="${1#*=}"; shift;;
    --private) PRIVATE=1; shift;;
    --create-test-pr) CREATE_TEST_PR=1; shift;;
    --help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "Error: --repo owner/repo required"
  usage
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repo. Initialize and retry."
  exit 2
fi

CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
echo "Local branch: $CURRENT_BRANCH"

if command -v gh >/dev/null 2>&1; then
  echo "Using gh CLI"
  if gh repo view "$REPO" >/dev/null 2>&1; then
    echo "Repo exists."
  else
    if [[ $PRIVATE -eq 1 ]]; then
      gh repo create "$REPO" --private --source=. --remote=origin --push --confirm || true
    else
      gh repo create "$REPO" --public --source=. --remote=origin --push --confirm || true
    fi
  fi
  git branch -M main || true
  git push -u origin main
  if [[ $CREATE_TEST_PR -eq 1 && -f .github/workflows/ci.yml ]]; then
    BRANCH="test-ci-missing-tests"
    git checkout -b "$BRANCH" || git checkout "$BRANCH"
    cp .github/workflows/ci.yml .github/workflows/ci.yml.bak.pushci || true
    perl -0777 -pe 's/\brun:\s*npm.*test[^\n]*\n/    run: echo "SKIP_TESTS"\n/gi' -i .github/workflows/ci.yml || true
    git add .github/workflows/ci.yml || true
    git commit -m "ci(test): simulate missing tests" || true
    git push -u origin "$BRANCH" || true
    gh pr create --title "CI test: missing tests" --body "Test PR" --base main --head "$BRANCH" || true
  fi
  echo "Done via gh"
  exit 0
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "gh not available and GITHUB_TOKEN not set. Install gh or export GITHUB_TOKEN."
  exit 3
fi

API="https://api.github.com"
OWNER="${REPO%%/*}"
NAME="${REPO##*/}"

resp=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $GITHUB_TOKEN" "$API/repos/$REPO")
if [[ "$resp" -ne 200 ]]; then
  if [[ $PRIVATE -eq 1 ]]; then
    curl -s -H "Authorization: token $GITHUB_TOKEN" -d "{\"name\":\"$NAME\",\"private\":true}" "$API/user/repos" >/dev/null
  else
    curl -s -H "Authorization: token $GITHUB_TOKEN" -d "{\"name\":\"$NAME\",\"private\":false}" "$API/user/repos" >/dev/null
  fi
fi

REMOTE="https://$GITHUB_TOKEN@github.com/$REPO.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git branch -M main || true
git push -u origin main

if [[ $CREATE_TEST_PR -eq 1 && -f .github/workflows/ci.yml ]]; then
  BRANCH="test-ci-missing-tests"
  git checkout -b "$BRANCH" || git checkout "$BRANCH"
  cp .github/workflows/ci.yml .github/workflows/ci.yml.bak.pushci || true
  perl -0777 -pe 's/\brun:\s*npm.*test[^\n]*\n/    run: echo "SKIP_TESTS"\n/gi' -i .github/workflows/ci.yml || true
  git add .github/workflows/ci.yml || true
  git commit -m "ci(test): simulate missing tests" || true
  git push -u origin "$BRANCH" || true
  PR=$(curl -s -H "Authorization: token $GITHUB_TOKEN" -d "{\"title\":\"CI test: missing tests\",\"head\":\"$BRANCH\",\"base\":\"main\",\"body\":\"Test PR\"}" "$API/repos/$REPO/pulls")
  echo "PR created: $(echo "$PR" | sed -n 's/.*\"html_url\": *\"\([^\"]*\)\".*/\1/p')"
fi

echo "Done. Repo pushed to: https://github.com/${REPO}"
