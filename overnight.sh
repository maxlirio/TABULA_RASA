#!/bin/bash
# Overnight orchestrator: finish run1 (already training -> apollo.pt), test+preserve it,
# then chain a bigger run2 -> apollo2.pt (kept separate so the working model is never lost).
cd ~/Developer/TABULA_RASA
L=overnight.log
echo "=== orchestrator started ===" >> "$L"

# 1) wait for run1 to finish
while ! grep -q "^DONE" mixed.log 2>/dev/null; do sleep 30; done
echo "" >> "$L"; echo "### RUN1 DONE (28M, 448/8/192) ###" >> "$L"
grep -E "iter 20000" mixed.log | tail -1 >> "$L"
cp -f apollo.pt apollo_run1.pt 2>/dev/null

echo "--- run1: rule generalization ---" >> "$L"
PYTHONPATH=$PWD python3 test_rules.py apollo.pt >> "$L" 2>&1
echo "--- run1: chat/commands/reward ---" >> "$L"
PYTHONPATH=$PWD python3 test_brain.py apollo.pt >> "$L" 2>&1

# 2) bigger run2 -> apollo2.pt (does NOT touch apollo.pt)
echo "" >> "$L"; echo "### RUN2 START (bigger: 512/8/192) ###" >> "$L"
PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 -u train_lm.py mixed apollo2.pt "Apollo2" 20000 8 512 8 192 8 > mixed2.log 2>&1
echo "### RUN2 DONE ###" >> "$L"
grep -E "iter 20000" mixed2.log | tail -1 >> "$L"
echo "--- run2: rule generalization ---" >> "$L"
PYTHONPATH=$PWD python3 test_rules.py apollo2.pt >> "$L" 2>&1
echo "--- run2: chat/commands/reward ---" >> "$L"
PYTHONPATH=$PWD python3 test_brain.py apollo2.pt >> "$L" 2>&1

echo "" >> "$L"; echo "=== orchestrator complete ===" >> "$L"
