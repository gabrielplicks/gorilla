import unicodedata



def remove_control_chars_and_pad(s: str) -> str:

    """

    Removes all control characters from a string and pads it with 16 spaces on both sides.



    Args:

        s: The input string.



    Returns:

        The input string with all control characters removed and padded with 16 spaces on both sides.

    """

    filtered_str = ''.join(c for c in s if unicodedata.category(c) != 'Cc')

    padded_str = " " * 16 + filtered_str + " " * 16



    return padded_str

