import re



def validate_string(text: str) -> bool:

    """Validates that a string contains only a-z, A-Z, and 0-9 characters.



    Args:

        text: The string to validate.



    Returns:

        True if the string contains only a-z, A-Z, and 0-9 characters, False otherwise.

    """

    pattern = r'^[a-zA-Z0-9]*$'

    result = re.match(pattern, text)

    return result is not None

