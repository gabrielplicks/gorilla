import re



def is_alphabetical_string(string: str) -> bool:

    """

    Checks if a given string consists of only alphabetical characters, returning True if it does,

    and False if there are any non-alphabetical characters.



    Args:

        string: The input string to be checked.



    Returns:

        True if the string consists of only alphabetical characters, False otherwise.

    """

    if string == "":

        return False

    return bool(re.match(r"^[a-zA-Z]+$", string))

