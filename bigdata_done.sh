#!/bin/bash
cd ~/Developer/TABULA_RASA
while ! grep -q "^DONE" mixed.log 2>/dev/null; do sleep 30; done
echo "=== BIG-DATA RUN DONE ==="
grep "best gen" mixed.log | tail -1
echo "peak gen during training:"; grep -oE "gen [0-9.]+" mixed.log | sort -t' ' -k2 -n | tail -1
echo; echo "### held-out rule generalization ###"
PYTHONPATH=$PWD python3 test_rules.py apollo.pt
echo; echo "### multi-rule ###"
PYTHONPATH=$PWD python3 -c "
from gm.chat import Chat
from gm.lm import load
c=Chat({'a':load('apollo.pt')},'a','/tmp/mr2.json')
for q in ['say apple instead of hello','whenever i say bye say zoom','hello','bye','hello']:
    print(' ',q,'->',c.reply(q))
"
echo; echo "### chat incl LONGER inputs (babble / understanding check) ###"
PYTHONPATH=$PWD python3 -c "
from gm.chat import Chat
from gm.lm import load
c=Chat({'a':load('apollo.pt')},'a','/tmp/lc.json')
for q in ['hello','how are you today','what do you like to do for fun','i had a really long day at work','do you think it will rain tomorrow','tell me a story','what is your name']:
    print(' you>',q,'\n  bot>',c.reply(q))
"
echo; echo "### structured still ok ###"
PYTHONPATH=$PWD python3 test_brain.py apollo.pt
