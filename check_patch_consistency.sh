#!/bin/bash
echo "=================================================="
echo "1. Find all files with patch_height/patch_width/stride_patch defaults"
echo "=================================================="
grep -rln "patch_height\s*=\|patch_width\s*=\|stride_patch\s*=" --include="*.py" . 2>/dev/null

echo ""
echo "=================================================="
echo "2. Values in generate_metadata_patch.py"
echo "=================================================="
grep -n "patch_height\s*=\s*[0-9]\|patch_width\s*=\s*[0-9]\|stride_patch\s*=\s*[0-9]" src/preprocess/generate_metadata_patch.py 2>/dev/null

echo ""
echo "=================================================="
echo "3. Values in sampling_patch_metadata.py"
echo "=================================================="
grep -n "patch_height\s*=\s*[0-9]\|patch_width\s*=\s*[0-9]\|stride_patch\s*=\s*[0-9]" src/preprocess/sampling_patch_metadata.py 2>/dev/null

echo ""
echo "=================================================="
echo "4. Values in any other preprocess script"
echo "=================================================="
grep -rn "patch_height\s*=\s*[0-9]\|patch_width\s*=\s*[0-9]\|stride_patch\s*=\s*[0-9]" src/preprocess/*.py 2>/dev/null | grep -v "generate_metadata_patch.py\|sampling_patch_metadata.py"

echo ""
echo "=================================================="
echo "5. Values used in training scripts (patch config, if any)"
echo "=================================================="
grep -rn "patch_height\s*=\s*[0-9]\|patch_width\s*=\s*[0-9]\|patch_size\s*=\s*[0-9]" train/multimodal/*.py train/radar/*.py 2>/dev/null | grep -v "def \|cfg\."

echo ""
echo "DONE."