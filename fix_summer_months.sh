#!/bin/bash
# Fixes summer month definitions to {6, 7, 8, 9} (JJAS) in the two
# files that were found using the wrong {6, 7, 8} (JJA) definition.

FILE1="src/preprocess/generate_test_data_full_image.py"
FILE2="src/preprocess/generate_metadata_multihorizon_season.py"

echo "=================================================="
echo "BEFORE"
echo "=================================================="
echo "--- $FILE1 ---"
grep -n "{6, 7, 8}\|{6, 7, 8, 9}" "$FILE1"
echo ""
echo "--- $FILE2 ---"
grep -n "SUMMER_MONTHS\s*=" "$FILE2"

cp "$FILE1" "$FILE1.bak"
cp "$FILE2" "$FILE2.bak"

echo ""
echo "=================================================="
echo "Applying fixes"
echo "=================================================="
sed -i 's/{6, 7, 8}/{6, 7, 8, 9}/g' "$FILE1"
sed -i 's/SUMMER_MONTHS = {6, 7, 8}/SUMMER_MONTHS = {6, 7, 8, 9}/' "$FILE2"

echo "Fixed. Backups saved as .bak"

echo ""
echo "=================================================="
echo "AFTER — verify"
echo "=================================================="
echo "--- $FILE1 ---"
grep -n "{6, 7, 8}\|{6, 7, 8, 9}" "$FILE1"
echo ""
echo "--- $FILE2 ---"
grep -n "SUMMER_MONTHS\s*=" "$FILE2"

echo ""
echo "=================================================="
echo "Cross-check: any remaining {6,7,8} without the 9, anywhere"
echo "=================================================="
grep -rn "{6, 7, 8}\b" --include="*.py" src/ train/ 2>/dev/null
echo "(If nothing printed above, no JJA-only definitions remain.)"