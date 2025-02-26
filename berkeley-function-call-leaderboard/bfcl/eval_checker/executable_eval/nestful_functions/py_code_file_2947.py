def remove_redundant_whitespaces(s: str) -> str:

    """

    Removes redundant whitespaces from a string.



    Args:

        s: The input string.



    Returns:

        The input string with redundant whitespaces removed.

    """

    words = s.split()

    words = filter(lambda word: word != '', words)

    return ' '.join(words)

