"""A thin starter primer — a few hundred very simple sentences using basic words, so the
bot doesn't begin from absolute nothing. This is NOT a pretrained model; it's a tiny
hand-written reader (like a child's first words) that gives the embedding space some
initial shape: animals cluster together, actions together, colors together, opposites
land near their pairs. Everything else it learns from you.
"""

SEED = """
a bird is an animal
a fish is an animal
a dog is an animal
a cat is an animal
a cow is an animal
a bird is a small animal
a fish lives in water
a bird can fly
a fish can swim
a dog can run
a cat can run
a dog can walk
a cat can walk
a cow can walk
birds fly in the sky
fish swim in the water
dogs run and walk
cats walk and run
a dog is a good animal
a cat is a good animal
animals can move
animals eat food
a bird eats food
a fish eats food
a dog eats food
run is fast
walk is slow
to run is to move fast
to walk is to move slow
stop is not go
go is not stop
go means move
stop means do not move
you can run
you can walk
you can stop
you can go
you can jump
you can sit
you can eat
you can sleep
i can run
i can walk
i can stop
run and walk are things you do
stop and go are things you do
red is a color
blue is a color
green is a color
yellow is a color
black is a color
white is a color
red and blue are colors
green and yellow are colors
the sky is blue
the grass is green
the sun is yellow
snow is white
night is black
an apple is red
a banana is yellow
good is not bad
bad is not good
a good dog is not a bad dog
yes is not no
no is not yes
you say yes or no
hot is not cold
cold is not hot
big is not small
small is not big
fast is not slow
slow is not fast
up is not down
day is not night
a big animal is not small
a small bird is not big
i am good
you are good
i am happy
a happy dog is good
water is wet
fire is hot
ice is cold
the sun is hot
a bird is fast
a fish is fast
a cow is big
a dog is small
a cat is small
a bird is small
yes is good
no is bad
to eat is good
food is good
to run is fast
to stop is to rest
"""


def seed_sentences():
    """The primer as a list of token lists."""
    out = []
    for line in SEED.strip().splitlines():
        toks = line.strip().split()
        if toks:
            out.append(toks)
    return out
