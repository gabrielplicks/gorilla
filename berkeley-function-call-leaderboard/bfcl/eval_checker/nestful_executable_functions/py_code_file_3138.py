from typing import List, Tuple



def split_and_lowercase(strings: List[str]) -> List[Tuple[str, str]]:

    """Splits a list of strings into prefixes and suffixes, converts the prefixes to lowercase, and replaces every _ with a space.



    Args:

        strings: A list of strings in the format "[prefix]_[suffix]".



    Returns:

        A list of tuples with lowercased prefixes and suffixes.

    """

    tuples = []

    for string in strings:

        prefix, suffix = string.split('_')

        tuples.append((prefix.lower(), suffix.replace('_', ' ')))

    return tuples

