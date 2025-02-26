def replace_is_with_was(input_string: str) -> str:

    """

    Replaces all occurrences of the substring "is" with "was" in a given string.



    Args:

        input_string: The input string to modify.



    Returns:

        The modified string with all occurrences of "is" replaced with "was".

    """

    words = input_string.split()

    for i, word in enumerate(words):

        if word == "is":

            words[i] = "was"

    return " ".join(words)

