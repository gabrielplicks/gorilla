from typing import List



def parse_comma_separated_integers(input_string: str) -> List[int]:

    """Parses a string of comma-separated integers and returns a list of valid integers.



    Args:

        input_string: A string of comma-separated integers.



    Returns:

        A list of valid integers.

    """

    integers_as_strings = input_string.split(',')

    valid_integers = []

    for integer_as_string in integers_as_strings:

        try:

            integer = int(integer_as_string)

        except ValueError:

            continue

        valid_integers.append(integer)

    return valid_integers

