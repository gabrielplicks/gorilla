from typing import Dict, List



def find_matching_words(sentence: str, vocabulary: Dict[str, str]) -> List[str]:

    """Finds the matching words in a sentence based on a vocabulary object.



    Args:

        sentence: The sentence to search for matching words.

        vocabulary: A dictionary where the keys are words and the values are the meaning of the words.



    Returns:

        A list of the words in `sentence` that match keys in `vocabulary`.

    """

    return [word for word in sentence.split() if word in vocabulary]

