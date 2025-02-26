from typing import Dict, List



def count_hash_frequencies(strings: List[str]) -> Dict[int, int]:

    """Counts the number of occurrences of each unique string's hash value.

    Args:

        strings: A list of strings.

    Returns:

        A dictionary where the keys are the hash values and the values are the number of occurrences.

    """

    hash_frequencies = {}

    for string in strings:

        hash_value = hash(string)

        if hash_value in hash_frequencies:

            hash_frequencies[hash_value] += 1

        else:

            hash_frequencies[hash_value] = 1

    return hash_frequencies

