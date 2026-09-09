#!/bin/bash
# Updates tier-5 threshold/weight from 35 -> 40 in all 10 training scripts
# (both the __init__ defaults and the print statements), to match the
# actual Config class value confirmed to be 40 (DWD Starkregen criteria).
# Also verifies the Config class itself so you can confirm it already
# says 40 (source of truth) before/after this fix.

FILES=(
    "train/multimodal/train_ConvLSTM.py"
    "train/multimodal/train_simVP.py"
    "train/multimodal/train_smaAt_UNet.py"
    "train/multimodal/train_earthformer.py"
    "train/multimodal/train_VPTR.py"
    "train/radar/train_ConvLSTM.py"
    "train/radar/train_simVP.py"
    "train/radar/train_smaAt_UNet.py"
    "train/radar/train_earthformer.py"
    "train/radar/train_VPTR.py"
)

echo "=================================================="
echo "STEP 0 -- Confirm Config class tier-5 value in each file (source of truth)"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -n "weight_threshold_5\s*=\|weight_value_5\s*=" "$f" | grep -v "self\.\|def "
    fi
done

echo ""
echo "=================================================="
echo "STEP 1 -- BEFORE (init defaults / print statements with 35)"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -n "weight_value_5=35\.0\|weight_threshold_5=35\.0\|3/7/15/25/35" "$f"
    fi
done

echo ""
echo "=================================================="
echo "STEP 2 -- Applying fixes (backups saved as .bak)"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        cp "$f" "$f.bak"
        sed -i 's/weight_threshold_5=35\.0/weight_threshold_5=40.0/g' "$f"
        sed -i 's/weight_value_5=35\.0/weight_value_5=40.0/g' "$f"
        sed -i 's#3/7/15/25/35#3/7/15/25/40#g' "$f"
        echo "Fixed: $f"
    fi
done

echo ""
echo "=================================================="
echo "STEP 3 -- AFTER (verify)"
echo "=================================================="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -n "weight_value_5=40\.0\|weight_threshold_5=40\.0\|3/7/15/25/40" "$f"
    fi
done

echo ""
echo "=================================================="
echo "STEP 4 -- Final sweep: any remaining '35' tied to tier-5, anywhere"
echo "=================================================="
grep -rn "weight_threshold_5=35\.0\|weight_value_5=35\.0\|3/7/15/25/35" train/ 2>/dev/null
echo "(If nothing printed above, no leftover 35 references remain.)"