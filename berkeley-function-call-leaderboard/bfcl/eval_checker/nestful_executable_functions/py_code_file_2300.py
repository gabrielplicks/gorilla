import re



def replace_the_with_a(text: str) -> str:

    """Replaces all occurrences of the word 'the' with 'a', regardless of the case.



    Args:

        text: The input string.



    Returns:

        A new string in which all occurrences of the word 'the' are replaced with 'a'.

    """

    return re.sub(r'\bthe\b', 'a', text, flags=re.IGNORECASE)

