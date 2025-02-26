from typing import List



def join_words_with_commas(words: List[str]) -> str:

    """Joins a list of words with commas, and adds 'and' before the last word if there are more than three words.



    Args:

        words: A list of words to join.



    Returns:

        A string with the words joined with commas.

    """

    if len(words) > 2:

        return ', '.join(words[:-1]) + ', and ' + words[-1]

    else:

        return ', '.join(words)

