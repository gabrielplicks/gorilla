import re



def format_string_or_none(string: str) -> Union[str, None]:

    """Formats a string by converting it to lower case, removing spaces and special characters, and checking if it is in a valid format.



    Args:

        string: The input string.



    Returns:

        The formatted string or None if the input string is empty or None.

    """

    if string is None or string == "":

        return None

    formatted_string = string.lower().replace(" ", "").replace("-", "").replace("_", "")

    if not re.match(r'^[a-z]+$', formatted_string):

        return "Error: Invalid format for string."

    return formatted_string

