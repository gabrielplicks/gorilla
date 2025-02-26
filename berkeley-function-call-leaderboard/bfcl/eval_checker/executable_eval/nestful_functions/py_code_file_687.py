from collections import defaultdict

from typing import Dict, List



def get_frequency(strings: List[str]) -> Dict[str, int]:

    """Calculates the frequency of each unique string in a list.



    Args:

        strings: A list of strings.



    Returns:

        A dictionary where the keys are the unique strings and the values are the corresponding frequencies.

    """

    frequency = defaultdict(int)

    for s in strings:

        frequency[s] += 1

    return frequency

