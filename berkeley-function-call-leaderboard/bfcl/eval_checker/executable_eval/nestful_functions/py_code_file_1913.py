import re



def alphanumeric_only(string: str) -> str:

    """Converts a string of mixed alphanumeric and special characters into a new string with only alphanumeric characters.



    The function retains the case and order of the characters, and strips out any special characters and spaces.



    Args:

        string: The input string to be converted.



    Returns:

        A new string with only alphanumeric characters.

    """

    alphanumeric_string = re.sub(r'[^A-Za-z0-9]', '', string)

    return alphanumeric_string

