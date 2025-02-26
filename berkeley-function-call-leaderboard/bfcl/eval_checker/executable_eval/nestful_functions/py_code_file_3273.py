from typing import List



class EmptyListError(Exception):

    pass



def list_average(numbers: List[float]) -> float:

    """Calculates the average of a list of numbers.

    Args:

        numbers: A list of numbers.

    Returns:

        The average of the numbers.

    Raises:

        EmptyListError: If the list is empty.

    """

    try:

        if not numbers:

            raise EmptyListError("List is empty")

        total = sum(numbers)

        length = len(numbers)

        return total / length

    except Exception:

        return 0.0

