from typing import Dict, List



def sum_unique_values(lst: List[int]) -> int:

    """Calculates the sum of all unique values in a list of integers.

    Args:

        lst: A list of integers.

    Returns:

        The sum of all unique values in the list.

    """

    counts: Dict[int, int] = {}  # Dictionary to keep track of counts

    for num in lst:

        counts[num] = counts.get(num, 0) + 1  # Increment count for each value

    return sum(num for num in counts if counts[num] == 1)  # Sum only the unique values

