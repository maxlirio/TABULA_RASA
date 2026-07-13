#!/usr/bin/env python3
"""Richer conversational + STORYTELLING data. Two problems this targets:

1. STORIES: the model can't tell a story because prep_stories.py paired "tell me a story" with
   RANDOM book paragraphs (no beginning/middle/end) — so it learned no story FORM. Here we generate
   many SHORT, COMPLETE, internally-consistent stories (one character, a want, an event, an action,
   a resolution) with heavy slot variety, so the model learns the SHAPE of a story, not one story.
2. FLUENCY: natural small-talk with CORRECT sentiment (negative input -> sympathetic reply), varied
   greetings, and common intents — the thin/rough everyday chatter.

Output: data/dialogue2/chat.txt  (USER:/BOT: blocks, blank-line separated — t4_run.py adds the ■ EOS)
This is TRAINING DATA (varied, recombined), not hardcoded replies. Keep it a real slice of the
corpus (append at a few x) but mind reward-design exposure stays >=15% — preflight B4 guards that.
"""
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- STORY GRAMMAR (consistent slots)
# (intro noun phrase, later reference, subject pronoun, possessive) — kept consistent within a story
CHARACTERS = [
    ("a little fox", "the fox", "she", "her"), ("a young boy named Sam", "Sam", "he", "his"),
    ("a curious robot", "the robot", "it", "its"), ("an old sailor", "the sailor", "he", "his"),
    ("a small rabbit", "the rabbit", "he", "his"), ("a girl named Mia", "Mia", "she", "her"),
    ("a lonely dragon", "the dragon", "she", "her"), ("a clever cat", "the cat", "he", "his"),
    ("a brave knight", "the knight", "she", "her"), ("a tiny mouse", "the mouse", "he", "his"),
    ("a tired farmer", "the farmer", "he", "his"), ("a kind old woman", "the woman", "she", "her"),
    ("a young owl", "the owl", "she", "her"), ("a wandering musician", "the musician", "he", "his"),
    ("a little turtle", "the turtle", "he", "his"), ("a star that fell to earth", "the star", "it", "its"),
]
SETTINGS = ["in a deep green forest", "by the edge of the sea", "in a quiet little village",
            "high on a windy mountain", "in a busy town", "in a garden full of flowers",
            "under an old oak tree", "beside a slow river", "in a snowy valley",
            "at the edge of a wide desert", "in a house at the end of the lane"]
EVENTS = ["a great storm rolled in from the north", "a quiet stranger arrived at dawn",
          "the only path split into two", "the sky turned a strange shade of gold",
          "a small voice called out from the trees", "the river rose higher than ever before",
          "an old map fell from a passing cart", "the last light of the day began to fade"]
COMPLICATIONS = ["the road was harder than expected", "no one would listen at first",
                 "the old bridge had broken in two", "the nights grew cold and long",
                 "fear whispered to turn back", "the map seemed to lead in circles"]
# QUESTS bundle want -> actions that PURSUE that want -> resolutions that SATISFY it. Picking the
# three slots independently produced incoherent stories (a knight who "wanted a sunflower" then
# "built a boat" and "saw the ocean"); the model dutifully learned that stories need not cohere.
QUESTS = [
    ("wanted more than anything to find a friend",
     ["asked everyone in the village for help", "set off down the long road",
      "followed the sound through the dark"],
     ["made a true friend along the way", "found someone just as lonely, and was lonely no more"]),
    ("dreamed of seeing the ocean one day",
     ["packed a bag and did not look back", "set off down the long road",
      "climbed the steep and rocky hill"],
     ["saw the ocean at last, wide and shining and blue",
      "stood at the shore and watched the waves come in"]),
    ("wanted to learn how to fly",
     ["climbed the steep and rocky hill", "tried again every morning until {pp} arms ached",
      "watched the birds for hours and copied them"],
     ["rose into the air at last, wobbling but flying",
      "learned that flying was mostly falling with hope"]),
    ("was searching for a lost golden key",
     ["opened the heavy wooden door", "followed the old map through the dark",
      "searched every room in the empty house"],
     ["found the key at the bottom of a dusty drawer",
      "found exactly what {ref} had been looking for"]),
    ("wanted to grow the tallest sunflower in the land",
     ["planted the seed in the best soil {pp} could find",
      "carried water up the hill every single morning",
      "sat very still and watched the small green shoot"],
     ["grew a sunflower taller than the rooftops",
      "learned that patience was the answer all along"]),
    ("longed to hear music again",
     ["followed the sound through the dark", "asked everyone in the village for help",
      "opened the heavy wooden door"],
     ["heard the old song ring out once more",
      "found the music had been waiting all along"]),
    ("wanted to be brave, just once",
     ["opened the heavy wooden door", "walked straight toward the thing {p} feared most",
      "packed a bag and did not look back"],
     ["found that the courage had been inside {pp} heart all along",
      "did the frightening thing, and found {p} could"]),
    ("hoped to find the way back home",
     ["followed the old map through the dark", "climbed the steep and rocky hill",
      "asked everyone in the village for help"],
     ["came home happier than ever before", "saw the lights of home at the end of the lane"]),
    ("wanted to see what lay beyond the hills",
     ["packed a bag and did not look back", "set off down the long road",
      "climbed the steep and rocky hill"],
     ["stood at the top and saw the whole wide world below",
      "found a valley no one had ever named"]),
    ("wished to make the village smile",
     ["asked everyone in the village for help", "worked in secret through the night",
      "sat very still and listened"],
     ["helped someone else, and felt {pp} own heart grow lighter",
      "made the whole village laugh out loud at last"]),
]
CLOSINGS = ["And from that day on, things were never quite so lonely.", "It was a good day, after all.",
            "And that is how it came to be.", "The end.",
            "And they say {ref} still tells the story to this day.", "Some things are worth the long road."]
STORY_REQ = ["tell me a story", "can you tell me a story", "tell me a little story",
             "i'd love to hear a story", "tell me a bedtime story", "make up a story for me",
             "spin me a tale", "tell me a short story", "read me a story", "story time!"]


def _cap(s):
    """Capitalize the first letter of every sentence (names already carry their own caps)."""
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.strip())


def _story(r):
    intro, ref, p, pp = r.choice(CHARACTERS)
    # connectors are set by the templates, so complications stay plain clauses (no leading "but")
    tmpl = r.choice([
        ("once upon a time, {intro} lived {setting}. {ref} {want}. "
         "one day, {event}, and so {p} {action}. but {comp}. "
         "still, {p} did not give up, and in the end {p} {res}. {close}"),
        ("long ago, {setting}, there lived {intro} who {want}. "
         "then one morning, {event}. without a word, {ref} {action}, though {comp}. "
         "{p} kept going, and at last {p} {res}. {close}"),
        ("there was once {intro} who {want}. every day was the same, until {event}. "
         "that was the day {ref} {action}. for a while {comp}, and {p} nearly gave up. "
         "but {p} tried once more, and {p} {res}. {close}"),
    ])
    want, actions, resolutions = r.choice(QUESTS)   # action + resolution FIT the want
    sent = tmpl.format(
        intro=intro, ref=ref, setting=r.choice(SETTINGS), p=p, pp=pp,
        want=want, event=r.choice(EVENTS),
        action=r.choice(actions).format(p=p, pp=pp),
        comp=r.choice(COMPLICATIONS),
        res=r.choice(resolutions).format(ref=ref, p=p, pp=pp),
        close=r.choice(CLOSINGS).format(ref=ref))
    return _cap(sent)


# ---------------------------------------------------------------- NATURAL CHAT (sentiment-correct)
# (list of user phrasings, list of good replies). Negative feelings ALWAYS map to sympathy.
CHAT = [
    (["hey", "hi", "hello", "heya", "yo", "hey there", "hi there", "howdy", "good morning",
      "good evening", "hey apollo"],
     ["hey! how's it going?", "hi there! what's up?", "hello! good to see you.",
      "hey! what's on your mind?", "hi! how are you doing today?"]),
    (["i'm anxious", "i feel anxious", "i'm worried", "i'm stressed", "i'm feeling stressed",
      "i'm overwhelmed", "i had a rough day", "i'm having a hard time", "i feel down",
      "i'm sad", "i feel sad today", "i'm upset", "i'm exhausted", "i can't sleep",
      "everything feels hard right now", "i'm struggling"],
     ["i'm sorry you're feeling that way. want to talk about it?",
      "that sounds really hard. i'm here if you want to talk.",
      "i'm sorry it's been rough. take it easy on yourself tonight.",
      "that's tough. do you want to tell me what's going on?",
      "i hear you. it's okay to have hard days - i'm right here."]),
    (["i'm happy", "i feel great", "i'm excited", "i had a wonderful day", "good things happened",
      "i got a promotion", "i passed my exam", "i'm in a great mood"],
     ["that's wonderful to hear!", "i'm so glad! what happened?", "that's fantastic - congratulations!",
      "love that. tell me more!", "amazing! you earned it."]),
    (["what can you do", "what are you able to do", "how can you help", "what do you do"],
     ["i can chat, remember things you teach me, do math, tell the time, and turn a goal into a "
      "reward spec. and i can try to tell you a story.",
      "i'm a little local model - i can talk, learn facts and word meanings you teach me, do "
      "arithmetic, and design rewards. want to try one?"]),
    (["thank you", "thanks", "thanks so much", "i appreciate it"],
     ["you're welcome!", "anytime!", "happy to help.", "of course!"]),
    (["that's not what i meant", "you misunderstood", "no, i meant something else"],
     ["sorry about that - can you say it a different way?", "my mistake. what did you mean?",
      "got it, let me try again - tell me more?"]),
    (["never mind", "forget it", "don't worry about it", "it's nothing"],
     ["okay, no problem.", "sure, whenever you're ready.", "alright - i'm here if you need me."]),
    (["how are you", "how are you doing", "how's it going", "how have you been"],
     ["i'm doing well, thanks! how about you?", "pretty good! how's your day?",
      "i'm good. what's new with you?"]),
    (["goodbye", "bye", "see you later", "good night", "i have to go", "talk later"],
     ["bye! take care.", "see you later!", "good night!", "talk soon - take care!"]),
    (["i'm bored", "i have nothing to do", "entertain me"],
     ["want me to tell you a story?", "you could teach me something new, or give me a goal to turn "
      "into a reward. or i can tell you a story!", "let's chat - or i can spin you a quick tale."]),
]


def main(n_story=6000, chat_reps=800, seed=41):
    # Stories come from BOTH sources on purpose. prep_stories2's real fables carry the variety, but
    # fables-only (run #1) left the model with no reliable ARC — asked for a story it recited fable
    # titles instead. These templates are the one source proven to yield complete beginning-middle-end
    # stories, so they go back in as the minority backbone; the fables supply the imagination.
    r = random.Random(seed)
    out = []
    for _ in range(n_story):
        out.append(f"USER: {r.choice(STORY_REQ)}\nBOT: {_story(r)}")
    for _ in range(chat_reps):
        for ways, reps in CHAT:
            out.append(f"USER: {r.choice(ways)}\nBOT: {r.choice(reps)}")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "dialogue2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[dialogue2] {n_story:,} short stories + {chat_reps * len(CHAT):,} chat turns "
          f"-> data/dialogue2/chat.txt ({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
