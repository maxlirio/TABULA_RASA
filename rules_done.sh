#!/bin/bash
cd ~/Developer/TABULA_RASA
while ! grep -q "^DONE" mixed.log 2>/dev/null; do sleep 20; done
echo "=== CORRECTED RUN DONE ==="
grep "best gen" mixed.log | tail -1
echo "(best generalization seen during training:)"
grep -oE "gen [0-9.]+" mixed.log | sort -t' ' -k2 -n | tail -1
echo; echo "### held-out rule generalization (final apollo.pt) ###"
PYTHONPATH=$PWD python3 test_rules.py apollo.pt
echo; echo "### chat / commands / reward ###"
PYTHONPATH=$PWD python3 test_brain.py apollo.pt
echo; echo "### multi-rule check ###"
PYTHONPATH=$PWD python3 -c "
from gm.chat import Chat
from gm.lm import load
c=Chat({'a':load('apollo.pt')},'a','/tmp/mr.json')
for q in ['say apple instead of hello','whenever i say bye say zoom','hello','bye','hello']:
    print(' ',q,'->',c.reply(q))
"
