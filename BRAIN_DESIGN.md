# The Brain That Grows Into Its Suit

**One line:** a portable brain you drop into a body. It arrives understanding *goals* but not
*how to reach them*. You tell it, in language, **what commands it has** — not what they do. It
then discovers what they do by doing them, and learns to **arrange** them to satisfy goals.

It **discovers** actions already at its disposal. It does not **invent** them.

The README in this repo describes the pre-LM project (embeddings + logic engine) and is stale.
This is the actual goal, written down because it kept drifting — including for its author.

---

## 1. Why the language model can't be the planner

The instinct is to ask the LM to decompose a goal into steps. We tested it (2026-07-14, run #1
weights). It cannot, and the failure is instructive:

```
"how do i move the boulder out of the way?"  -> "A flooded underpass is in the way..."
"roll the boulder aside"                     -> "A boulder is in the way. i could find a log
                                                 or bridge, wade through..."   (river actions)
"find a lever"                               -> "A lever is a stability and a stability."
```

It does not decompose. It pattern-matches the prompt's *shape* to a memorized template and fills
the slots — losing the object entirely (boulder -> shoelace). Nothing in the corpus teaches
`command -> sub-commands`, so no such operator exists in the weights.

But the deeper reason is not fixable with data, and this is the load-bearing claim of this design:

> **Text cannot teach you what your arm does.** It can only teach you what people *say* about arms.
> What `push` does *in this suit* is not a fact about English. It is a fact about the suit, and the
> only way to get it is to push and look.

"A lever is a stability and a stability" is not a bug. It is a model that has read the word *lever*
and has never once pushed on one. Scaling the LM does not fix this; it only makes the confabulation
more fluent. (SayCan works because a 540B model read the internet — that is a *different* solution,
and it is the one this project's from-scratch rule forbids.)

## 2. The four parts

| part | job | learns from |
|---|---|---|
| **Mouth** (the LM) | parse a goal from language; take the action *names*; explain back | text |
| **Tendrils** (world model) | what does each command *do*? `f(o, a) -> o'` | **interaction** |
| **Arranger** (planner) | search the learned model for a sequence reaching the goal | the tendrils |
| **Suit** | exposes an opaque command list + observations | — it just is |

**The mouth does not plan.** It is the interface, and only the interface — the reason the brain
talks at all is so the operator doesn't have to write code. That is its entire job.

**The arranging is search, not language.** Once `f` is learned, a plan is found by rolling the
model forward and searching for an action sequence that reaches the goal. The intelligence in
"arranging" lives in search over a learned model. It never needed a language model.

## 3. What is given vs. what is discovered

This is the line the whole project stands on:

- **GIVEN:** the *names* of the commands. `push`, `grab`, `step_left`. A list of opaque handles.
- **DISCOVERED:** what every one of them *does*. Their effects, their preconditions, which are
  reversible, which do nothing, which are the same action under two names.

A classical PDDL planner is handed `push(X): pre: reachable(X); eff: not blocked(path)` — a human
pre-chewing the understanding. That is exactly what this design refuses. The brain gets the handle
and must earn the meaning.

## 4. The learning signal is surprise, not reward

The brain guesses what a command will do, does it, and is wrong. **Prediction error is the
gradient.** No reward function, no reward tokens, no human-designed scoring — self-supervised
from the world pushing back.

Goals are then satisfied by *search toward a described state*, not by maximizing a scalar. That is
the difference between a brain that understands a goal and a machine that wants a number, and it is
the point of the whole exercise.

## 5. The unresolved fork (state it, don't hide it)

`SELF_MOD_BRAIN/DESIGN.md` says: *"the core is already a capable reasoner (the LLM core)... NOT
learning to be intelligent from scratch."* TABULA_RASA's founding rule is: **no pretrained LLM,
ever.** These are incompatible, and the project has been quietly running both.

**We do not have to resolve it to make progress**, and this is the useful part:

> The tendrils train on `(o, a, o')` tuples. No text. No priors. No world knowledge.
> **The world-model layer is valid under either bet.**

So build the tendrils and the arranger first. They are fork-independent. And the experiment will
*measure* how much the core actually needs to know — which is a far better basis for deciding the
fork than arguing about it.

## 6. The three tests (no robot, no scale, laptop-sized)

The thesis is falsifiable. These are the numbers:

1. **Does it learn the suit?** Prediction error must *drop*. If it doesn't, there are no tendrils.
   Guard against self-deception: measure on *held-out* transitions, not the babble it trained on.
2. **Does it arrange, or memorize?** It must reach goals **it never practiced** — novel goal states,
   requiring action sequences never seen during babbling. This is the difference between a plan and
   a lookup table, and this project has already been fooled by a memorized lookup once (reward
   design). Assume it will be again unless the test forbids it.
3. **Is it portable?** **Swap the suit** — rename the commands, permute them, rewire which handle
   does what — and the brain must relearn *without* retraining the language side. If a permutation
   of the action names breaks it, it never grew tendrils; it just memorized a keyboard.

Test 3 is the actual thesis. A brain that survives a suit-swap is a brain that grew into a suit,
rather than one that was born knowing it.

## 7. Honest ceiling

- This is a **general learner specializing to one host** — not general intelligence, and it will not
  become that by accumulating skills. Same honest ceiling `SELF_MOD_BRAIN/DESIGN.md` already states.
- The tendrils give it *this* suit. They give it nothing about the next one, except a method.
- The mouth stays weak. A 285M from-scratch model will not become a good planner, and this design
  does not ask it to. It asks it to hear a goal and name a command — which is roughly the ceiling of
  what it demonstrably can do.
- **What would falsify this:** if the planner can only reach goals whose action sequences appeared in
  the babble, then "arranging" is a lookup and the whole thesis is dead. Test 2 exists to catch that,
  and it must be run honestly.

## 8. Order of work

1. **Suit:** a rule-flip gridworld (from `SELF_MOD_BRAIN/DESIGN.md`). Opaque command handles.
2. **Tendrils:** babble, learn `f(o, a) -> o'` from prediction error, on held-out transitions.
3. **Arranger:** search the learned model to hit a goal state. Cache what works (as `gm/solver.py`
   already does for sequences — that pattern is correct and should be reused).
4. **Mouth:** wire the LM in for `language -> goal state` only. Last, and least.
5. **Suit-swap:** permute the action handles. Re-run. This is the thesis test.
6. Only then: the real suit (`ROBOT_EXPERIMENT`).
