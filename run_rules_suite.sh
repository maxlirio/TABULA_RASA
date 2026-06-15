#!/bin/bash
cd ~/Developer/TABULA_RASA
for i in $(seq 1 2400); do grep -q "^DONE" mixed.log 2>/dev/null && break; sleep 8; done
echo "=================== TRAINING DONE ==================="
grep -E "iter 9000" mixed.log | tail -1
echo; echo "############### RULE-FOLLOWING GENERALIZATION ###############"
PYTHONPATH=$PWD python3 test_rules.py
echo; echo "############### still chats / structured ok? ###############"
PYTHONPATH=$PWD python3 test_brain.py apollo.pt
