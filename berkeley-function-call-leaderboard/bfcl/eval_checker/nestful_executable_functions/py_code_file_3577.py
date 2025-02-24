import re

from typing import List



def tokenization_filter(text: str, min_length: int) -> List[str]:

    """Performs tokenization on a given input text and filters out words that are shorter than a specified minimum length.



    Args:

        text: The input text to be tokenized.

        min_length: The minimum length of words to be included in the output.



    Returns:

        A list of lowercase words from the input text that are longer than or equal to the specified minimum length.

    """

    tokens = re.split(r"\s|\t|\n|[^\w\s]", text)

    filtered_tokens = [token.lower() for token in tokens if len(token) >= min_length]



    return filtered_tokens

