from typing import List



def target_number_exists(numbers: List[int], target: int) -> bool:

    """Checks if the target number exists in the list of numbers.



    Args:

        numbers: A list of integers.

        target: The target integer to find in the list.



    Returns:

        True if the target number is found in the list, False otherwise.

    """

    found = False

    for num in numbers:

        if num == target:

            found = True

            break



    return found

