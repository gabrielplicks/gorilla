from typing import Dict, List, Union



def create_word_count_dict(words: List[str], default_value: Union[int, None] = None) -> Dict[str, Union[int, None]]:

    """Creates a dictionary of unique words mapped to their counts or a default value.



    Args:

        words: A list of words (strings).

        default_value: An optional default value to use instead of the frequency count.



    Returns:

        A dictionary of unique words mapped to their counts or the default value.

    """

    if not words:

        return {}

    word_counts = {}

    for word in words:

        if word in word_counts:

            word_counts[word] += 1

        else:

            word_counts[word] = 1

    if default_value is not None:

        word_counts = {word: default_value for word in word_counts}



    return word_counts

