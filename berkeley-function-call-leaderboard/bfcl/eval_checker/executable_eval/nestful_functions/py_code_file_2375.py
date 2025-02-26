from typing import List



def concatenate_sentence(words: List[str], separator: str) -> str:

    """Concatenates a list of words into a single sentence using a separator character.



    Args:

        words: A list of strings representing the words in the sentence.

        separator: A string representing the separator character.

    """

    return separator.join(words)

