from typing import List, Set



def can_sum_to_target(numbers: List[int], target: int) -> bool:

    """Determines whether there are two numbers in the list whose sum equals the target value.



    Args:

        numbers: A list of numbers.

        target: The target value.



    Returns:

        A boolean value indicating whether the target value can be achieved by summing two numbers from the list.

    """

    seen: Set[int] = set()

    for num in numbers:

        if target - num in seen:

            return True

        seen.add(num)

    return False

