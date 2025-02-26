from collections import Counter

from typing import List



def histogram_distance(list1: List[int], list2: List[int]) -> int:

    """Calculates the histogram distance between two lists.

    The histogram distance is computed as the sum of the absolute differences between the frequencies of each element in the two lists.

    Args:

        list1: The first list.

        list2: The second list.

    """

    freq1 = Counter(list1)

    freq2 = Counter(list2)

    combined_elements = set(list1) | set(list2)  # Combine unique elements from both lists

    dist = 0

    for elem in combined_elements:

        dist += abs(freq1[elem] - freq2[elem])

    return dist

