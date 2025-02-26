from typing import Dict, List



def get_total_frequency(frequency: Dict[str, int], keys: List[str]) -> int:

    """Calculates the total frequency of a set of keys in a frequency dictionary.



    Args:

        frequency: A dictionary where keys are items and values are their frequencies.

        keys: A list of keys to evaluate.



    """

    total_frequency = 0

    for key in keys:

        if key in frequency:

            total_frequency += frequency[key]

        else:

            frequency[key] = 0

    return total_frequency

