#!/bin/bash
echo "=================================================="
echo "1. Files mentioning 'summer' (case-insensitive)"
echo "=================================================="
grep -rl "summer" --include="*.py" . 2>/dev/null | grep -v "__pycache__"

echo ""
echo "=================================================="
echo "2. Month-based summer definitions (looking for {6,7,8,9} or similar)"
echo "=================================================="
grep -rn "month not in\|month in\|{6, *7, *8, *9}\|\[6, *7, *8, *9\]\|(6, *7, *8, *9)" --include="*.py" . 2>/dev/null | grep -v "__pycache__"

echo ""
echo "=================================================="
echo "3. Any OTHER month combinations that might define summer differently"
echo "   (e.g. JJA only = 6,7,8 without 9)"
echo "=================================================="
grep -rn "{6, *7, *8}\|\[6, *7, *8\]\|(6, *7, *8)\b" --include="*.py" . 2>/dev/null | grep -v "__pycache__"

echo ""
echo "=================================================="
echo "4. Filenames containing 'summer' (data files, may hint at definition)"
echo "=================================================="
find . -iname "*summer*" 2>/dev/null | grep -v "__pycache__"

echo ""
echo "DONE."