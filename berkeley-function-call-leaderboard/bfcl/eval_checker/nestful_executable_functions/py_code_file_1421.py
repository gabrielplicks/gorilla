import re



def is_valid_hex_color(color_code: str) -> bool:

    """Checks if a given string is a valid hexadecimal color code.



    Args:

        color_code: The string to check.



    Returns:

        True if the string is a valid hexadecimal color code, False otherwise.

    """

    pattern = r"^#[a-fA-F0-9]{6}$"

    return bool(re.match(pattern, color_code))

