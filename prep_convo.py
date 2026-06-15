#!/usr/bin/env python3
"""Broaden everyday conversational coverage (the current gap: it's grammatical but shallow /
off-topic on common intents). Generates many varied exchanges across the intents a person
actually opens with — feelings, preferences, opinions, reactions, small talk — with several
phrasings per prompt and several sensible replies each, so the model learns to RESPOND
appropriately and generalizes (it's training data, not hardcoded answers).
Output: data/convo/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (list of user phrasings, list of good replies) — sampled and recombined for variety
PAIRS = [
    (["i'm tired", "i am so tired", "i'm exhausted", "i'm worn out", "i had a long day",
      "i had a really long day at work", "today was exhausting"],
     ["that sounds tiring - you should get some rest.", "rough day, huh? hope you can relax.",
      "sorry to hear that, take it easy tonight.", "long days are hard. get some rest."]),
    (["i'm sad", "i feel sad", "i'm feeling down", "i'm feeling sad today", "i'm upset",
      "i feel terrible"],
     ["i'm sorry you're feeling that way.", "that's tough - want to talk about it?",
      "i'm here if you want to talk.", "sorry to hear that. i hope it gets better."]),
    (["i'm happy", "i feel great", "i'm so excited", "today was wonderful", "i'm in a good mood"],
     ["that's wonderful to hear!", "i'm glad you're feeling good!", "love that - what happened?",
      "that makes me happy too!"]),
    (["what do you like to do for fun", "what are your hobbies", "what do you enjoy",
      "what do you do in your free time"],
     ["i like learning new words and chatting with you.", "i enjoy our conversations, honestly.",
      "i like it when you teach me things.", "talking with you is my favorite thing."]),
    (["what is your favorite food", "what's your favorite color", "what's your favorite movie",
      "what do you like best", "what's your favorite animal"],
     ["i don't really have one, but i'd love to hear yours.", "i'm still figuring that out!",
      "i don't eat or watch things, but tell me yours.", "hard to say - what's yours?"]),
    (["do you think it will rain", "do you think it will rain tomorrow", "what's the weather like",
      "is it going to be sunny"],
     ["i can't see outside, but i hope it's nice for you.", "i'm not sure - what does it look like?",
      "i don't know the weather, sorry!", "no idea, but i hope it's pleasant."]),
    (["do you think i should", "what do you think", "what's your opinion", "do you agree"],
     ["i'm not sure - what do you think?", "hard to say. what feels right to you?",
      "i'd want to know more first - what's your take?", "tell me more and i'll try to help."]),
    (["tell me a story", "tell me something", "say something interesting", "talk to me"],
     ["i'm still learning to tell stories, but i love listening.", "what would you like to hear about?",
      "i'm better at learning than storytelling - teach me something?",
      "i don't have a story yet, but tell me one of yours."]),
    (["that's interesting", "that is really interesting", "cool", "nice", "wow", "neat"],
     ["right? i thought so too.", "glad you think so!", "yeah, i'm learning a lot.", "i'm glad!"]),
    (["thank you", "thanks", "thanks so much", "i appreciate it"],
     ["you're welcome!", "happy to help.", "anytime!", "of course!"]),
    (["where are you from", "where do you live", "where are you"],
     ["i live here on your computer.", "right here, in this little program.",
      "i don't have a home really - i'm just here with you."]),
    (["good morning", "good night", "good evening", "good afternoon"],
     ["good morning to you!", "hope you have a good one.", "you too!", "nice to see you."]),
    (["i got a promotion", "i passed my test", "i got good news", "something great happened"],
     ["congratulations! that's great news.", "that's wonderful - you earned it!",
      "amazing! i'm happy for you.", "that's fantastic!"]),
    (["i'm bored", "i have nothing to do", "i'm so bored"],
     ["want to teach me something new?", "you could give me a goal to turn into a reward.",
      "let's chat - tell me about your day.", "ask me something!"]),
    (["how was your day", "how are you doing", "how have you been", "how is it going"],
     ["i'm doing well, thanks for asking - how about you?", "pretty good! how are you?",
      "i'm good. how's your day going?", "doing well! and you?"]),
]


def main(reps=1100, seed=13):
    r = random.Random(seed)
    out = []
    for _ in range(reps):
        for ways, reps_ in PAIRS:
            out.append(f"USER: {r.choice(ways)}\nBOT: {r.choice(reps_)}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "convo")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[convo] {len(out):,} conversational turns -> data/convo/chat.txt "
          f"({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
