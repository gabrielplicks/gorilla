from typing import AnyStr



def check_string_validity(string: AnyStr) -> bool:

    """Checks if a string contains at least one uppercase letter, one lowercase letter, and one digit.



    Args:

        string: The input string.



    Returns:

        True if the string contains at least one uppercase letter, one lowercase letter, and one digit.

        False otherwise.

    """

    return any(char.isupper() for char in string) and \

           any(char.islower() for char in string) and \

           any(char.isdigit() for char in string)

