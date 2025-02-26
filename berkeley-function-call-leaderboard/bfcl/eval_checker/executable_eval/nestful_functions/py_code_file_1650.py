from typing import List



def convert_list_to_formatted_string(numbers: List[int]) -> str:

    """Converts a list of numbers into a formatted string.



    Each number is represented as a string, padded to 3 characters with leading zeros.

    If the number is even, the string is in uppercase. Otherwise, the string is in lowercase.

    The strings are separated by commas and a space.



    Args:

        numbers: A list of numbers.



    Returns:

        A formatted string.

    """

    result = []

    for number in numbers:

        string = str(number).zfill(3)

        if number % 2 == 0:

            string = string.upper()

        else:

            string = string.lower()

        result.append(string)

    return ', '.join(result)

