from typing import Union



def validate_header(header: str) -> bool:

    """Validates the header format of a text file containing a list of floating-point numbers.



    Args:

        header: The header string to validate.



    Returns:

        True if the header format is valid, False otherwise.

    """

    try:

        num_lines, num_digits_after_decimal = header.split()

        num_lines = int(num_lines)

        num_digits_after_decimal = int(num_digits_after_decimal)

        return len(header.split()) == 2 and num_lines > 0 and num_digits_after_decimal >= 0

    except ValueError:

        return False

