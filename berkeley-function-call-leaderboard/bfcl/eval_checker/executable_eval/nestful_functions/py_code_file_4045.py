from typing import List



def filter_long_strings(strings: List[str], threshold: int) -> List[str]:

    """Filters a list of strings based on their length.

    Args:

        strings: A list of strings.

        threshold: An integer threshold.

    Returns:

        A list of strings that are longer than or equal to the threshold.

    """

    return list(filter(lambda s: len(s) >= threshold, strings))

