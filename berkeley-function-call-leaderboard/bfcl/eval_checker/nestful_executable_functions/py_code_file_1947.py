from typing import Dict, List



def hash_words(words: List[str]) -> Dict[int, str]:

    """Creates a dictionary with the hash of each string as the key and the string itself as the value.



    Args:

        words: A list of strings.



    Returns:

        A dictionary with the hash of each string as the key and the string itself as the value.

    """

    return {hash(word): word for word in words}

