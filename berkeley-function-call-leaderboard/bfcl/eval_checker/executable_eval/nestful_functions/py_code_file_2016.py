def count_sequence(s: str, char: str) -> str:

    """

    Counts the number of occurrences of `char` in `s`.

    Args:

        s: The string to search in.

        char: The character to search for.

    """

    count = 0

    for c in s:

        if c == char:

            count += 1



    if count >= 5:

        return str(count)

    else:

        return '0'

