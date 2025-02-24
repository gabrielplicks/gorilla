import random

random.seed(42)
from typing import List



def random_words(sentence: str, num_words: int) -> str:

    """Returns a new sentence with `num_words` random words from `sentence`, in the same order as they appear in the original sentence.



    Args:

        sentence: The original sentence.

        num_words: The number of words to select.

    """

    words: List[str] = sentence.split()

    selected_words: List[str] = random.sample(words, num_words)

    return " ".join(selected_words)

