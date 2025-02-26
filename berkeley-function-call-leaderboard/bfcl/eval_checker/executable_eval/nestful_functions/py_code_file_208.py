from typing import List



def parse_comma_separated_list(comma_separated_string: str) -> List[int]:

    """Parses a comma-separated string into a list of integers.

    Args:

        comma_separated_string: A string containing comma-separated integers.

    Returns:

        A list of integers parsed from the input string.

    """

    numbers = []

    substrings = comma_separated_string.split(',')

    for substring in substrings:

        try:

            number = int(substring)

            numbers.append(number)

        except ValueError:

            pass

    return numbers

