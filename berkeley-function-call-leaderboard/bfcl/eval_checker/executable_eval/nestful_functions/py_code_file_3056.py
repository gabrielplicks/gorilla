import re



def find_zip_code(input_string: str) -> str:

    """Extracts and returns the first five-digit sequence from a string.

    If the string does not contain a valid zip code, the function returns None.

    Args:

        input_string: The input string to search for a zip code.

    """

    zip_code_regex = r"\b[0-9]{5}\b"

    match = re.search(zip_code_regex, input_string)

    if match:

        return match.group()

    else:

        return None

