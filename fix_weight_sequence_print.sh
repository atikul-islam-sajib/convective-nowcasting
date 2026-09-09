#!/bin/bash
# Fixes the leftover incorrect weight sequence in the print statements of
# ConvLSTM and SimVP (both multimodal and radar) -- these still showed
# "3/8/40/80/120x" instead of the correct "3/7/15/25/40x" (weight = threshold),
# which was missed by the earlier bulk fix.

FILES=(
    "train/multimodal/train_ConvLSTM.py"
    "train/multimodal/train_simVP.py"
    "train/radar/train_ConvLSTM.py"
    "train/radar/train_simVP.py"
)

echo "=================================================="
echo "BEFORE"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -n "3/8/40/80/120x" "$f"
    fi
done

echo ""
echo "=================================================="
echo "Applying fix"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        cp "$f" "$f.bak2"
        sed -i 's#3/8/40/80/120x#3/7/15/25/40x#g' "$f"
        echo "Fixed: $f"
    fi
done

echo ""
echo "=================================================="
echo "AFTER -- verify"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -n "3/7/15/25/40x\|3/8/40/80/120x" "$f"
    fi
done

echo ""
echo "=================================================="
echo "Final sweep -- any '3/8/40/80/120' left anywhere in train/"
echo "=================================================="
grep -rn "3/8/40/80/120" train/ --include="*.py" 2>/dev/null
echo "(If nothing printed above -- other than .bak files -- the fix is complete.)"