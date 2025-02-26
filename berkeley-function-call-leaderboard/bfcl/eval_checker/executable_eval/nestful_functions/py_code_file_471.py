from typing import List



class CountResult:

    def __init__(self, total: int, is_all_odd: bool):

        self.total = total

        self.is_all_odd = is_all_odd



def count_and_check(integers: List[int]) -> CountResult:

    """

    Calculates the total sum of a list of integers and checks if all integers are odd.

    Args:

        integers: A list of integers.

    Raises:

        ValueError: If any integer is not odd.

    """

    if any(num % 2 == 0 for num in integers):

        raise ValueError('Not all integers are odd')



    total = sum(integers)

    is_all_odd = all(num % 2 != 0 for num in integers)

    return CountResult(total, is_all_odd)

