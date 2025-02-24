from typing import List



def concatenate_word(word: str, n: int, words: List[str]) -> List[str]:

    """Concatenates `n` number of `word`s to each string in `words`.



    Args:

        word: The string to be concatenated.

        n: The number of times `word` should be concatenated.

        words: The list of strings to which `word` will be concatenated.



    Returns:

        A new list of strings with `word` concatenated `n` times to each string in `words`.

        If `n` is less than or equal to zero, then the function returns an empty list.

    """

    if n <= 0 or not isinstance(words, list) or not all(isinstance(w, str) for w in words):

        return []

    new_words = []

    for w in words:

        new_words.append(w + (word * n))

    return new_words

