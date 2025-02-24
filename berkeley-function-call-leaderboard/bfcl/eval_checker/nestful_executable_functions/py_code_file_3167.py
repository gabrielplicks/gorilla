from typing import List



def extract_ngrams(text: str, n: int) -> List[str]:

    """Extracts all n-grams from a given text.



    Args:

        text: The input text.

        n: The size of the n-gram to extract.



    Raises:

        ValueError: If n is greater than the length of the text.

    """

    if n > len(text):

        raise ValueError("n cannot be greater than the length of the text")

    ngrams = []

    for i in range(len(text) - n + 1):

        ngram = text[i:i+n]

        ngrams.append(ngram)

    return ngrams

