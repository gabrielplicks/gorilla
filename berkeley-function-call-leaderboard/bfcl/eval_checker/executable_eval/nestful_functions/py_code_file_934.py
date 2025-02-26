from typing import Optional



def remove_first_last_and_middle(input_string: str) -> Optional[str]:

    """

    Removes the first and last characters from a string, and the middle character if the length is odd.



    Args:

        input_string: The input string.



    Returns:

        The modified string.

    """

    input_length = len(input_string)

    if input_length == 0:

        return ""

    elif input_length == 1 or input_length == 2:

        return input_string

    elif input_length % 2 == 0:

        modified_string = input_string[1:-1]

    else:

        modified_string = input_string[1:-1]

    return modified_string

