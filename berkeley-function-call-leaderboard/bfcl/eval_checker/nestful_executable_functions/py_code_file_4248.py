import re



def replace_hello_with_bye(text: str) -> str:

    """Replaces all occurrences of the string "hello" with "bye" in a given string.



    Args:

        text: The input string.



    Returns:

        A new string with all occurrences of "hello" replaced with "bye".

    """

    return re.sub(r"hello", "bye", text)

