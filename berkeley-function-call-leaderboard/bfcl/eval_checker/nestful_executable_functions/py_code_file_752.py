from typing import List, Tuple



def find_overlapping_tuples(tuples: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """Finds the overlapping tuples in a list of tuples.



    Args:

        tuples: A list of tuples, each containing a start and end time.



    Returns:

        A list of tuples that overlap.

    """

    tuples.sort(key=lambda x: x[0])  # Sort the tuples by start time

    overlapping_tuples = []

    for i in range(len(tuples) - 1):

        current_tuple = tuples[i]

        next_tuple = tuples[i + 1]

        if current_tuple[1] > next_tuple[0]:  # Check if the tuples overlap

            overlapping_tuples.append(next_tuple)

    return overlapping_tuples

