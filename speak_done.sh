#!/bin/bash
cd ~/Developer/TABULA_RASA
while ! grep -q "^DONE" mixed.log 2>/dev/null; do sleep 30; done
echo "=== SPEAKING ROUND DONE ==="; grep "best gen" mixed.log | tail -1
echo; echo "### chat quality (the bar) ###"
PYTHONPATH=$PWD python3 -c "
from gm.chat import Chat
from gm.lm import load
c=Chat({'a':load('apollo.pt')},'a','/tmp/sd.json')
qs=['hello','how are you today','what do you like to do for fun','i had a really long day at work','i am feeling sad today','what is your favorite food','do you think it will rain tomorrow','tell me a story','i got a promotion today','that is really interesting','where are you from','goodbye','my dog is sick','what should i eat for dinner','i love sunny days']
for q in qs: print(' you>',q,'\n  bot>',c.reply(q))
"
