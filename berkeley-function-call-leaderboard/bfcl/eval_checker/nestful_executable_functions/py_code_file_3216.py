from typing import List



def repeated_elements(lst: List[int]) -> bool:

    """Checks if any element is repeated in a list of integers.



    Args:

        lst: A list of integers.



    Returns:

        True if any element is repeated, False otherwise.

    """

    unique_elements = set()

    for element in lst:

        if element in unique_elements:

            return True

        unique_elements.add(element)

    return False

