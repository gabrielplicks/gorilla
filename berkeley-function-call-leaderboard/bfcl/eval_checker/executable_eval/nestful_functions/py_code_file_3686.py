import json

from typing import Dict, Set



def create_json_word_count(text: str) -> str:

    """Creates a JSON object with the number of occurrences of each word in a given text.



    Args:

        text: The input text.



    Returns:

        A JSON object with the number of occurrences of each word in the input text.



    Raises:

        ValueError: If the input is not a string.

    """

    if not isinstance(text, str):

        raise ValueError('Input must be a string')



    word_count: Dict[str, int] = {}

    seen_words: Set[str] = set()



    for word in text.split():

        if word not in seen_words:

            word_count[word] = 1

            seen_words.add(word)

        else:

            word_count[word] += 1



    return json.dumps(word_count)

