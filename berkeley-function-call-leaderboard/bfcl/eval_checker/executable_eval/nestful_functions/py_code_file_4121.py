from typing import List



def generate_numbers(output_even: bool = False) -> List[int]:

    """Generates numbers from 1 to 31, inclusive, and returns them in a list.

    If the optional `output_even` parameter is set to `True`, only even numbers are included in the list.

    Args:

        output_even: If `True`, only even numbers are included in the list.

    Returns:

        A list of numbers from 1 to 31, or a list of only the even numbers from 1 to 31.

    """

    return [i for i in range(1, 32) if (not output_even or i % 2 == 0)]

