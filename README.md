# TABULA_RASA

A talking bot that starts as a blank slate. You **talk to it and teach it**, and it learns
your words two different ways — and tells you when the things you teach it don't add up.

```
$ python3 talk.py
Hello! You can teach me words and phrases, and I will remember them...
> your name is Sprout
Thank you! I'll be Sprout from now on. Nice to meet you!
> what is bird like?
'bird' feels similar to: cat, color, fish, dog, animal.   (rough early on — "color" is noise)
> light is a word
> dark is when there is no light
> night is when it is dark and there are stars
I don't know what star is. What is star?
> stars are light
Wait — that doesn't add up. A night could never actually exist — a night would have to be
both light and not light. ... Which did you mean?
```

## Two ways it learns — and no LLM

There's no large language model in here. Two small home-grown parts, working together:

1. **Learning by usage** (`gm/embed.py`) — a word-embedding model written from scratch in
   numpy (skip-gram with negative sampling, hand-written gradients, no pretrained weights).
   Every sentence you type trains it. It learns *which words are alike from how you use them* —
   so it figures out that "bird" patterns like "fish" and "cat" without ever being told. This
   is the part that gives it a feel for *when/how* a word is used, not just a definition.

2. **Learning by definition** (`gm/abstract.py`) — a tiny logic engine. You can teach it ideas
   in words ("a bird is an animal", "dark is when there is no light"), and it reasons over them:
   it checks them for **consistency**, catches **contradictions**, names the exact conflicting
   quality, and even **derives facts you never stated** ("stars are light" + "dark is no light"
   ⟹ "a star is never dark").

The neural side is fuzzy and learned; the logic side is exact and explainable. Together they're
a small mind that learns language from you and keeps it coherent.

On top of those two it also:

- **Understands commands** — say "run", "please jump", or teach a new one ("you can hop") and it
  maps your words to an action it can do (exactly, or by closest meaning) and says it's doing it.
- **Asks the part of speech first** — meet a new word and it asks whether it's a noun, verb,
  adjective, adverb, pronoun, preposition, conjunction, or interjection, then asks the right
  follow-up. (It can't tell on its own — so it asks.)
- **Forms its own thoughts** — tell it "a robin is a bird" and it reasons further by itself:
  "I know a bird can fly — so can a robin fly too?" — then learns from your answer. It also asks
  its own questions from gaps in what it knows, sometimes unprompted.
- **Takes correction** — "that's wrong", "X is not like Y", "you used X wrong" → it fixes it.
- **Separates the message from the words** (`gm/say.py`) — what it means to say is a structured
  *message*; a separate language brain turns it into words. Teach it "say soar for fly" and it
  re-words the same thought ("a robin can soar"). The wording even varies with the content.

It begins with a **thin primer** (`gm/corpus.py`) — a few dozen toddler-simple sentences
(animals, actions, colors, opposites) so it isn't starting from absolute nothing. Everything
else it picks up from talking with you, and remembers in a brain file across sessions.

## Things to try

```
your name is X                       # give it a name (it remembers)
a cat is an animal                   # teach a fact
a cat can run                        # teach what it can do
a cat has whiskers                   # teach its parts
run / please jump / you can hop      # command it (and teach new commands)
what is cat like?                    # what it's learned cat patterns like
is a cat like a dog?                 # similarity it learned from usage
what are you curious about?          # a question it generates from its own gaps
say leap for jump                    # change the word it uses for a meaning
is a robin an animal?               # reasoning over your facts
dark is when there is no light       # define an idea
can there be a night?                # ask whether your definitions are satisfiable
tell me about cat                    # blends what it feels + what you've told it
forget cat                           # unteach a word
what do you know?                    # how much it's picked up
```

Honest about limits: trained only on what you type, the embeddings are **rough**, especially
early — it will make wrong guesses, and it can't write fluent novel sentences. It gets better
the more you talk to it. The *exact* understanding comes from the logic side; the neural side
adds *intuition*.

## Running it

```
python3 talk.py                            # talk to it
PYTHONPATH=$PWD python3 selftest.py        # dialogue + reasoning + embedding tests
PYTHONPATH=$PWD python3 fuzz_abstract.py   # property fuzzer for the logic engine
PYTHONPATH=$PWD python3 fuzz_bot.py        # crash fuzzer for the conversation
```

A definition also teaches the neural side: tell it "a robin is a bird" and "what is robin
like?" immediately answers with bird-ish words — the symbolic and neural halves reinforce
each other. You can also teach synonyms ("big means large") and it'll guess-and-check new
words it half-recognizes ("Is glip a bit like bird?").

## Layout

- `gm/embed.py` — from-scratch skip-gram word embeddings (numpy).
- `gm/corpus.py` — the thin starter primer.
- `gm/abstract.py` — the consistency / contradiction engine (kept from the project's roots).
- `gm/mind.py` — the bot's whole state (name, learned words, ideas, corpus), persisted.
- `gm/agent.py` — the conversation: naming, small talk, similarity, reasoning, learning.
- `talk.py` — the REPL.

*(Earlier this project was a blocks-on-a-table experiment; that's been retired in favor of the
talking bot, but the reasoning engine it grew is still the logical heart of this one.)*
