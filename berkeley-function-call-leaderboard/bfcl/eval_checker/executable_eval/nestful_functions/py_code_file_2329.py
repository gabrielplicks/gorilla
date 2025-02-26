from typing import List



def comma_separated_string(numbers: List[int]) -> str:

    """Converts a list of numbers into a string with each number separated by a comma and a space.



    Args:

        numbers: A list of numbers.



    Returns:

        A string with each number separated by a comma and a space.

    """

    result = []



    for number in numbers:

        result.append(str(number))



    return ', '.join(result)

