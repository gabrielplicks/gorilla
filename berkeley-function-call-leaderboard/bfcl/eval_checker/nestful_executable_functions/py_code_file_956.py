from typing import List



def generate_even_numbers_list() -> List[int]:

    """Generates a list of even numbers from 2 to 100 using a list comprehension.



    Returns:

        A list of even integers from 2 to 100.

    """

    return [num for num in range(2, 101) if num % 2 == 0]

