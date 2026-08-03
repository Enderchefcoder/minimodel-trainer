"""A tiny corpus bundled with the package.

Every stage of the pipeline - tokenizer training, pretraining, SFT, CoT, DPO and
RLVR - needs *some* data to be exercised. Shipping a small offline corpus means
the test suite, the smoke test and a first-time user's `minimodel quickstart` all
work with no network access and no license questions.

The text is original, deliberately simple, and repetitive in structure so that a
model with a few hundred thousand parameters can visibly learn it within a few
hundred steps. It is a fixture, not a real training corpus.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUILTIN_COT",
    "BUILTIN_PREFERENCE",
    "BUILTIN_RLVR",
    "BUILTIN_SFT",
    "BUILTIN_TEXT",
    "builtin_records",
]

BUILTIN_TEXT: list[str] = [
    "The river runs east through the valley. In spring the water is high and cold, "
    "and the stones along the bank are dark and smooth. Fish move in the shallows "
    "where the current slows.",
    "A small model learns slowly at first. It sees a word, then another word, and "
    "guesses what comes next. When the guess is wrong it changes a little. After "
    "many words the guesses become better.",
    "Bread needs flour, water, salt and time. Mix the flour and water and let the "
    "dough rest. Add the salt, then fold the dough every half hour. The longer the "
    "rest, the better the bread.",
    "The library opens at nine and closes at six. On Sunday it opens later. The "
    "reading room is on the second floor, above the entrance hall, where the light "
    "comes in from the tall windows.",
    "To measure the height of a tree, stand where you can see the top. Hold a "
    "stick upright at arm's length. Walk back until the stick covers the tree. "
    "The distance you walked is close to the height of the tree.",
    "Winter came early that year. The road was covered by noon and the buses "
    "stopped running. People walked instead, slowly, along the middle of the "
    "street where the snow was thinnest.",
    "A map is not the ground. It leaves things out on purpose, because a map that "
    "kept everything would be as large as the place it described and just as hard "
    "to read.",
    "The cat sleeps in the window in the afternoon. When the sun moves, the cat "
    "moves. By evening it has crossed the whole room without ever seeming to walk.",
    "Salt draws water out of vegetables. Slice the cucumber, add salt, and wait ten "
    "minutes. Pour off the water that collects. The slices will stay crisp instead "
    "of turning soft in the dressing.",
    "Every language has words it cannot translate cleanly. This is not a flaw. It "
    "means the language grew around a set of things its speakers needed to say "
    "often, and those things differ from place to place.",
    "The train leaves at seven and arrives at ten. It stops twice, once at the "
    "junction and once at the old mill town, where nobody gets off any more.",
    "Rope is stronger when it is dry. Wet fibres slide against each other and the "
    "knot loosens. Sailors learned this early, which is why they used knots that "
    "tighten under load rather than knots that only look secure.",
    "Counting is older than writing. The first marks people cut into bone were "
    "tallies, one line for one thing, and only much later did anyone think to give "
    "the number itself a name.",
    "The garden needs sun in the morning and shade in the afternoon. Plant the "
    "beans along the east fence and the lettuce under the apple tree, where the "
    "leaves keep the ground cool.",
    "A good question is worth more than a quick answer. The answer closes the "
    "subject; the question opens it. Ask what would have to be true for the answer "
    "to be wrong.",
    "Iron rusts because it prefers to be an oxide. Left alone with water and air it "
    "returns, slowly, to something close to the ore it was dug from. Paint does not "
    "stop this; it only delays it.",
    "The clock in the hall runs two minutes fast. Everyone in the house knows this "
    "and nobody has fixed it. They have all learned to subtract instead.",
    "Sound travels faster in water than in air, and faster still in steel. Put your "
    "ear to the rail and you will hear the train long before you hear it through "
    "the air.",
    "Old paths follow the easiest ground, not the shortest line. They bend around "
    "wet places and steep places. A road built later may cut straight through, but "
    "the path is usually the pleasanter walk.",
    "The recipe says to rest the meat after cooking. This is not fussiness. The "
    "juices are under pressure and will run out if you cut too soon, leaving the "
    "meat dry and the board wet.",
    "Two people can carry a long board more easily than one, but only if they walk "
    "in step. Out of step, each fights the other's swing, and the board feels "
    "heavier than it did alone.",
    "Words drift. A word that meant one thing to your grandparents may mean "
    "something narrower now, or wider, or nearly the opposite. Dictionaries record "
    "the drift; they do not stop it.",
    "The lighthouse flashes twice, then waits. Every light on the coast has its own "
    "pattern, so that a sailor who cannot see the shore can still tell which "
    "stretch of it he is looking at.",
    "Wood splits along the grain because the fibres run that way. Split with the "
    "grain and the axe does most of the work. Split across it and you are cutting, "
    "not splitting, and it will take all afternoon.",
    "A list is easier to remember than a paragraph, and a story is easier to "
    "remember than a list. This is why instructions that matter are so often told "
    "as something that happened to someone.",
    "The bridge was built in three parts and met in the middle within an inch. The "
    "engineers had allowed for the sun: the steel is longer on a hot day, and they "
    "chose a cool morning to close the gap.",
    "Bees fly further in dry weather. When the air is damp they stay near the hive, "
    "and the hum from the entrance is lower and more constant, as if the whole "
    "colony were waiting for something.",
    "Practice is not repetition. Repetition is doing the same thing again; practice "
    "is doing it again while paying attention to what went wrong, which is harder "
    "and much less pleasant.",
    "The kettle boils faster with less water in it, which seems obvious, and yet "
    "most people fill it to the top out of habit and then wait, and then pour most "
    "of it away.",
    "Maps of the sea show depth, not height. The numbers get smaller as the danger "
    "gets greater, which is the opposite of the way land maps work, and has misled "
    "more than one new sailor.",
    "Grammar is what a language does when nobody is watching. The rules in the book "
    "were written down afterwards, by people who listened carefully to what "
    "speakers already did without thinking.",
    "A wall built without mortar can stand for centuries if each stone rests on two "
    "below it. The skill is not in the lifting but in the choosing: finding the "
    "stone whose shape the gap already wants.",
]

BUILTIN_SFT: list[dict[str, Any]] = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"instruction": "Say hello.", "output": "Hello! How can I help you today?"},
    {
        "instruction": "Explain what a tokenizer does.",
        "output": "A tokenizer splits text into small pieces called tokens and maps each one to a number, so a model can work with text as a sequence of integers.",
    },
    {
        "instruction": "Give me three uses for salt in cooking.",
        "output": "1. Seasoning, to bring out flavour.\n2. Drawing water out of vegetables before they are dressed.\n3. Preserving food, by making it inhospitable to bacteria.",
    },
    {
        "instruction": "Summarise this in one sentence.",
        "input": "The river runs east through the valley. In spring the water is high and cold, and the stones along the bank are dark and smooth.",
        "output": "An eastward river through a valley runs high and cold in spring, with dark smooth stones along its bank.",
    },
    {"instruction": "What is 12 plus 30?", "output": "12 plus 30 is 42."},
    {
        "instruction": "How do I keep bread from going stale?",
        "output": "Keep it in a paper bag or a bread box at room temperature. Do not refrigerate it, because cold speeds up the staling process. For longer storage, slice it and freeze it.",
    },
    {
        "instruction": "Write a two-line poem about winter.",
        "output": "The road went white before the noon bell rang.\nWe walked the middle, where the snow was thin.",
    },
    {"instruction": "hey", "output": "Hey! What's up?"},
    {"instruction": "thanks", "output": "You're welcome."},
    {
        "instruction": "What is the difference between a map and the ground it shows?",
        "output": "A map leaves things out on purpose. A map that recorded everything would be as large and as confusing as the place itself, so every map is a choice about what matters.",
    },
    {
        "instruction": "Why does iron rust?",
        "output": "Iron reacts with oxygen and water to form iron oxide, which is more chemically stable than pure iron. Paint slows the reaction by keeping air and water off the metal, but it does not stop it.",
    },
]

BUILTIN_COT: list[dict[str, Any]] = [
    {
        "instruction": "A train leaves at 7 and arrives at 10. How long is the journey?",
        "reasoning": "The train leaves at 7 and arrives at 10. From 7 to 10 is 3 hours.",
        "output": "The journey takes 3 hours.",
    },
    {
        "instruction": "I have 5 apples and buy 8 more. Then I give away 3. How many do I have?",
        "reasoning": "Start with 5. Buying 8 more gives 5 + 8 = 13. Giving away 3 leaves 13 - 3 = 10.",
        "output": "You have 10 apples.",
    },
    {
        "instruction": "A board is 12 feet long. I cut it into 4 equal pieces. How long is each?",
        "reasoning": "12 feet divided into 4 equal pieces is 12 / 4 = 3 feet each.",
        "output": "Each piece is 3 feet long.",
    },
    {
        "instruction": "The clock in the hall is 2 minutes fast. It reads 9:15. What is the real time?",
        "reasoning": "The clock is ahead by 2 minutes, so the real time is 9:15 minus 2 minutes, which is 9:13.",
        "output": "The real time is 9:13.",
    },
    {
        "instruction": "If a kettle holds 2 litres and I need 500 ml, how many cups of 250 ml can I fill?",
        "reasoning": "I need 500 ml. Each cup holds 250 ml. 500 / 250 = 2 cups.",
        "output": "You can fill 2 cups.",
    },
]

BUILTIN_PREFERENCE: list[dict[str, Any]] = [
    {
        "prompt": "Say hello.",
        "chosen": "Hello! How can I help you today?",
        "rejected": "hello hello hello hello hello",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France is a country in Europe with many cities.",
    },
    {
        "prompt": "What is 12 plus 30?",
        "chosen": "12 plus 30 is 42.",
        "rejected": "That is a math question about addition.",
    },
    {
        "prompt": "How do I keep bread fresh?",
        "chosen": "Store it in a paper bag at room temperature, and freeze it if you need to keep it longer.",
        "rejected": "Bread is made from flour and water and salt and yeast.",
    },
    {
        "prompt": "hey",
        "chosen": "Hey! What's up?",
        "rejected": "Greetings. I am an artificial intelligence language model trained to assist you with a wide variety of tasks.",
    },
]

BUILTIN_RLVR: list[dict[str, Any]] = [
    {"prompt": "What is 5 plus 8?", "answer": "13"},
    {"prompt": "What is 12 plus 30?", "answer": "42"},
    {"prompt": "What is 20 minus 7?", "answer": "13"},
    {"prompt": "What is 6 times 7?", "answer": "42"},
    {"prompt": "What is 12 divided by 4?", "answer": "3"},
    {"prompt": "What is 9 plus 9?", "answer": "18"},
    {"prompt": "What is 100 minus 58?", "answer": "42"},
    {"prompt": "What is 3 times 5?", "answer": "15"},
]


def builtin_records(stage: str = "pretrain", *, repeat: int = 1) -> list[dict[str, Any]]:
    """Return the bundled records for a training ``stage``.

    Parameters
    ----------
    stage:
        One of ``pretrain``, ``sft``, ``cot``, ``preference`` or ``rlvr``.
    repeat:
        Tile the records this many times. Useful for tokenizer training, where
        BPE needs pair frequencies above the minimum threshold.

    >>> len(builtin_records("rlvr")) > 0
    True
    """
    tables: dict[str, list[Any]] = {
        "pretrain": [{"text": text} for text in BUILTIN_TEXT],
        "text": [{"text": text} for text in BUILTIN_TEXT],
        "sft": BUILTIN_SFT,
        "cot": BUILTIN_COT,
        "preference": BUILTIN_PREFERENCE,
        "rlvr": BUILTIN_RLVR,
    }
    if stage not in tables:
        raise ValueError(f"unknown builtin stage {stage!r}; available: {sorted(tables)}")
    records = tables[stage]
    return [dict(record) for _ in range(max(1, int(repeat))) for record in records]
