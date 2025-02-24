import re



def generate_regex_pattern(input_str: str) -> re.Pattern:

    """Generates a regular expression pattern that matches a string with exactly 20 characters,

    where the first 10 are digits and the last 10 are alphabets.



    Args:

        input_str: The input string. This is not used in the function.



    Returns:

        A compiled regular expression pattern.

    """

    pattern = re.compile("^[0-9]{10}[A-Za-z]{10}$")

    return pattern

