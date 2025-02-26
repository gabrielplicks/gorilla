from typing import List, Tuple



def extract_unique_elements(numbers: List[int]) -> List[Tuple[int, int]]:

    """Extracts all the unique elements and their corresponding positions in the list.

    Args:

        numbers: A list of integers.

    Returns:

        A list of tuples, each containing the element and its position in the original list.

    """

    unique_elements = []

    for index, num in enumerate(numbers):

        if num not in [n for n, _ in unique_elements]:

            unique_elements.append((num, index))

    return unique_elements

