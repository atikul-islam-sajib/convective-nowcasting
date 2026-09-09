#!/usr/bin/env bash
# fix_weight5_mismatch.sh
#
# Finds and fixes the constructor-default mismatch:
#   weight_threshold_5=35.0, weight_value_5=35.0
# which should match the module-level config value:
#   weight_threshold_5 = 40.0; weight_value_5 = 40.0
#
# Usage:
#   ./fix_weight5_mismatch.sh          -> dry run, shows affected files/lines
#   ./fix_weight5_mismatch.sh --fix    -> actually edits the files

set -uo pipefail

TRAIN_DIR="train"
FIX=false
[[ "${1:-}" == "--fix" ]] && FIX=true

OLD='weight_threshold_5=35.0, weight_value_5=35.0'
NEW='weight_threshold_5=40.0, weight_value_5=40.0'

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Error: '$TRAIN_DIR' directory not found. Run from the project root." >&2
  exit 1
fi

echo "Searching for mismatched defaults in '$TRAIN_DIR'..."
echo "-----------------------------------------------------"

matches=$(grep -rln --include="*.py" -F "$OLD" "$TRAIN_DIR" || true)

if [[ -z "$matches" ]]; then
  echo "No mismatches found. Everything already consistent."
  exit 0
fi

count=0
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  n=$(grep -cF "$OLD" "$file")
  count=$((count + n))
  echo "FOUND ($n occurrence(s)): $file"
  grep -nF "$OLD" "$file"
  echo ""

  if $FIX; then
    sed -i "s/${OLD}/${NEW}/g" "$file"
    echo "  -> fixed"
  fi
done <<< "$matches"

echo "-----------------------------------------------------"
if $FIX; then
  echo "Fixed $count occurrence(s) across $(echo "$matches" | wc -l) file(s)."
  echo "Run again without --fix to confirm nothing is left."
else
  echo "Total: $count occurrence(s) in $(echo "$matches" | wc -l) file(s)."
  echo "Re-run with --fix to correct them."
fi