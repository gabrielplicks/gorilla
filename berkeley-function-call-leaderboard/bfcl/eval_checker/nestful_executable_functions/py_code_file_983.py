from typing import List



def generate_ngrams(sequence: List[int], n: int) -> List[List[int]]:

    """Generates all possible n-grams (subsequences of length `n`) from `sequence`.



    Args:

        sequence: The input sequence.

        n: The length of the n-grams.



    Returns:

        A list of n-grams.

    """

    if len(sequence) < n:

        return []

    ngrams = []

    for i in range(len(sequence) - n + 1):

        ngrams.append(sequence[i:i+n])

    return ngrams

