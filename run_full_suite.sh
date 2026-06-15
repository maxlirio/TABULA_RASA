#!/bin/bash
cd ~/Developer/TABULA_RASA
for i in $(seq 1 2000); do grep -q "^DONE" mixed.log 2>/dev/null && break; sleep 8; done
echo "=================== TRAINING DONE ==================="
grep -E "iter 8000" mixed.log | tail -1
echo; echo "################## 3-WAY (chat/command/reward) ##################"
PYTHONPATH=$PWD python3 test_brain.py apollo.pt
echo; echo "################## VOCABULARY ##################"
PYTHONPATH=$PWD python3 -c "
from gm.chat import Chat
from gm.lm import load
c=Chat({'apollo':load('apollo.pt')},'apollo','/tmp/vocab.json')
c.know.triples=[]; c.notes=[]; c.history=[]
for q in ['what does free mean?','define honest','what does brave mean?','what does ocean mean?','how are you today?','that is a clever idea']:
    print('you>',q,'\n  bot:',c.reply(q))
import os; os.remove('/tmp/vocab.json')
"
echo; echo "################## REASONING + KNOWLEDGE ##################"
PYTHONPATH=$PWD python3 test_reason.py apollo.pt
