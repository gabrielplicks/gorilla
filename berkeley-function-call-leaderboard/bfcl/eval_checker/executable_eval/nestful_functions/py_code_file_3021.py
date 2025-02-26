from typing import Union



def update_string(input_string: Union[str, None]) -> str:

    """Updates a string by capitalizing the first letter of each word.



    Args:

        input_string: The input string to be updated.



    Returns:

        The updated string with the first letter of each word capitalized.

    """

    if not input_string.strip():

        return ""



    words = input_string.split()

    return " ".join(word.title() for word in words)

