import random

random.seed(42)
from typing import List



VERBS: List[str] = ["walked", "ran", "jumped"]

ADVERBS: List[str] = ["quickly", "slowly", "noisily"]

ADJECTIVES: List[str] = ["big", "small", "tall"]

NOUNS: List[str] = ["dog", "cat", "bird"]



def generate_random_phrase() -> str:

    """Generates a random verbal phrase based on the template: "She (verb) (adverb) the (adjective) (noun)."



    Args:

        None



    Returns:

        A string containing the generated phrase.

    """

    verb = random.choice(VERBS)

    adverb = random.choice(ADVERBS)

    adjective = random.choice(ADJECTIVES)

    noun = random.choice(NOUNS)

    return f"She {verb} {adverb} the {adjective} {noun}."

