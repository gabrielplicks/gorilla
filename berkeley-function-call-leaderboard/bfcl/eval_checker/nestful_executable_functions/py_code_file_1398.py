import re



def extract_string_between_brackets(text: str) -> str:

    """Extracts the string enclosed within brackets if it is found, or a default value if not.



    Args:

        text: The input string.



    Returns:

        The string enclosed within brackets if it is found, or the default value "No match found".

    """

    match = re.search(r"\[(.*?)\]", text)

    if match:

        return match.group(1)

    else:

        return "No match found"

