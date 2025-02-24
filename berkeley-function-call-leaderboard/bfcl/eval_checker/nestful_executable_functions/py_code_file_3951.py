from typing import AnyStr



def validate_alphanumeric_string(string: AnyStr) -> bool:

    """Validates an input string of alphanumeric characters.

    The input string must contain at least 1 lowercase letter, 1 uppercase letter, and 1 number.

    Args:

        string: The input string to validate.

    """

    return any(char.isalnum() for char in string) and \

        any(char.islower() for char in string) and \

        any(char.isupper() for char in string) and \

        any(char.isdigit() for char in string)

