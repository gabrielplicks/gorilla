from typing import Dict, List



def occurrences_fraction(lst: List[int]) -> Dict[int, float]:

    """Calculates the fraction of times each element appears in a list of integers.



    Args:

        lst: A list of integers.



    Returns:

        A dictionary that maps each element in the list to the fraction of the times it appears in the list.

    """

    occurrence_counts = {}

    for element in lst:

        if element in occurrence_counts:

            occurrence_counts[element] += 1

        else:

            occurrence_counts[element] = 1

    fraction_counts = {}

    total_occurrences = len(lst)

    for element, count in occurrence_counts.items():

        fraction_counts[element] = count / total_occurrences



    return fraction_counts

