#!/usr/bin/env python3
"""Consistent chit-chat frames so the brain is pleasant to talk to for the COMMON turns.

The movie-dialogue corpus is too high-variety for a tiny model to chat reliably, and the
strong command/reward frames make it read plain greetings as commands ("hello" -> a command).
The fix is the same trick that made commands/rewards work: make the everyday conversational
exchanges CONSISTENT frames too, repeated with varied phrasings, so a greeting is reliably
answered as a greeting. Output: data/chitchat/chat.txt (USER:/BOT: turns).
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# each entry: list of ways to say it -> list of friendly replies (picked at random)
PAIRS = [
    (["hi", "hello", "hey", "hiya", "yo", "hey there", "hello there", "hi there"],
     ["hello!", "hi there!", "hey!", "hi! good to see you.", "hello! how are you?"]),
    (["how are you", "how are you doing", "how's it going", "how do you do",
      "how are things", "you ok", "are you ok"],
     ["i'm good, thanks!", "doing well - you?", "pretty good! how about you?",
      "all good here.", "i'm well, thank you."]),
    (["what is your name", "who are you", "what should i call you", "what's your name"],
     ["you can call me apollo.", "i'm apollo.", "i'm a small brain you can talk to.",
      "apollo, nice to meet you."]),
    (["what are you", "are you a robot", "are you a person", "what kind of thing are you"],
     ["i'm a tiny brain you can talk to.", "just a little learning brain.",
      "i'm a small model, still learning."]),
    (["what can you do", "what do you do", "how do i use you", "help"],
     ["you can chat with me, give me a command, or give me a goal and i'll make a reward.",
      "talk to me, or tell me a goal and i'll turn it into a reward.",
      "give me a command or a goal, or just chat."]),
    (["thank you", "thanks", "thanks a lot", "cheers", "ty"],
     ["you're welcome!", "anytime!", "no problem!", "happy to help."]),
    (["do you like me", "are we friends", "do you care"],
     ["of course!", "yes, i like talking with you.", "we're friends!"]),
    (["good morning", "good evening", "good afternoon"],
     ["good morning!", "hello! good to see you.", "hi! hope you're well."]),
    (["how old are you", "when were you made"],
     ["i'm brand new.", "pretty young - still learning.", "just born, really."]),
    (["where are you", "where do you live"],
     ["right here, in your terminal.", "here with you.", "i live on your computer."]),
    (["yes", "yeah", "yep", "sure", "ok", "okay"],
     ["okay!", "got it.", "great.", "sounds good."]),
    (["no", "nope", "not really"],
     ["okay.", "no problem.", "alright.", "understood."]),
    (["nice to meet you", "good to meet you"],
     ["nice to meet you too!", "likewise!", "good to meet you!"]),
    (["i'm bored", "i am bored", "entertain me"],
     ["want to give me a goal to turn into a reward?", "tell me to do something!",
      "ask me anything."]),
    (["you are smart", "you're smart", "good job", "well done", "nice"],
     ["thank you!", "thanks - i'm learning.", "that's kind of you."]),
]


def main(n=700, seed=4):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        for ways, replies in PAIRS:
            out.append(f"USER: {r.choice(ways)}\nBOT: {r.choice(replies)}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "chitchat")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[chitchat] {len(out):,} turns -> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
