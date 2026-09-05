#!/bin/bash
# Updates the default values of p99_top_ratio (r=0.40) and prob (p=0.60)
# in the pre-sampling script, so the code matches what is reported in
# the thesis. Also fixes the stale help-text strings that showed the
# wrong old defaults (0.30 / 0.90).
#
# Run from the directory containing your presample script, or edit
# TARGET_FILE below to point at its actual path.

TARGET_FILE="src/preprocess/sampling_patch_metadata.py"

echo "=================================================="
echo "BEFORE — current default values"
echo "=================================================="
grep -n "p99_top_ratio\s*=\|prob\s*=\|default: 0\." "$TARGET_FILE"

cp "$TARGET_FILE" "$TARGET_FILE.bak"

echo ""
echo "=================================================="
echo "Applying changes"
echo "=================================================="

# Fix the function default arguments
sed -i 's/p99_top_ratio=0\.35/p99_top_ratio=0.40/' "$TARGET_FILE"
sed -i 's/prob=0\.65/prob=0.60/' "$TARGET_FILE"

# Fix the argparse defaults
sed -i "s/default=0\.35,\$/default=0.40,/" "$TARGET_FILE"
sed -i "s/default=0\.65,\$/default=0.60,/" "$TARGET_FILE"

# Fix the stale help-text numbers so they match the new real defaults
sed -i 's/default: 0\.30/default: 0.40/' "$TARGET_FILE"
sed -i 's/default: 0\.90/default: 0.60/' "$TARGET_FILE"

echo "Done. Backup saved at $TARGET_FILE.bak"

echo ""
echo "=================================================="
echo "AFTER — updated default values (verify below)"
echo "=================================================="
grep -n "p99_top_ratio\s*=\|prob\s*=\|default: 0\." "$TARGET_FILE"