from typing import List



def if_divisible_by_five(numbers: List[int]) -> List[str]:

    """Creates a list of strings that are either 'Not divisible by five' or 'Divisible by five' based on the input list of integers.



    Args:

        numbers: A list of integers.



    Returns:

        A list of strings that are either 'Not divisible by five' or 'Divisible by five'.

    """

    return list(map(lambda x: 'Divisible by five' if x % 5 == 0 else 'Not divisible by five', numbers))

