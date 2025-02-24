from typing import List



def parse_hexadecimal(input_str: str) -> str:

    """Converts a string containing a list of hexadecimal numbers separated by commas to a string of comma-separated integers.



    Args:

        input_str: A string containing a list of hexadecimal numbers separated by commas.



    Returns:

        A string of comma-separated integers in the order of their corresponding hexadecimal numbers in the input string.

    """

    return ','.join(str(int(num, base=16)) for num in input_str.split(','))

