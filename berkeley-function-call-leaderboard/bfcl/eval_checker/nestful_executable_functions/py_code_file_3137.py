def second_occurrence_index(s: str, c: str) -> int:

    """

    Returns the index of the second occurrence of a character in a string.

    If there is no second occurrence, returns -1.

    If the character occurs only once, returns -1.



    Args:

        s: The input string.

        c: The character to find.



    Returns:

        The index of the second occurrence of the character in the string, or -1 if there is no second occurrence.

    """

    count = s.count(c)

    if count < 2:

        return -1

    first_index = s.find(c)

    second_index = s.index(c, first_index + 1)



    return second_index

