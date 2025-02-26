from collections import defaultdict

from typing import Dict, List



def group_by_digit_frequency(numbers: List[int]) -> Dict[int, List[int]]:

    """Groups the elements of a list of numbers by their digit frequency.



    Args:

        numbers: A list of numbers.



    Returns:

        A dictionary where each key is the digit frequency and each value is the list of numbers with the given frequency.



    Raises:

        ValueError: If the provided list is empty.

    """

    if len(numbers) == 0:

        raise ValueError('The provided list is empty')

    frequencies = defaultdict(list)

    for number in numbers:

        digit_count = len(str(number))

        frequencies[digit_count].append(number)

    return frequencies

