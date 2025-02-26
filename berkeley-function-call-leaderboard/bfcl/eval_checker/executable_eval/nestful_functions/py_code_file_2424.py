from typing import List



def parse_list_of_numbers(string: str) -> List[float]:

    """Parses a string representing a list of numbers (separated by commas) into a list of floats.

    Args:

        string: The input string.

    Returns:

        A list of floats.

    """

    try:

        elements = string.split(',')

        numbers = [float(element) for element in elements]

        return numbers

    except ValueError:

        print('Invalid input: non-numerical characters or missing commas')

        return []

