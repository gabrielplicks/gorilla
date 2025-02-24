from typing import List



def split_with_escape(s: str, sep: str = ',', escape_char: str = '\\') -> List[str]:

    """Splits a string `s` using a separator `sep` and an escape character `escape_char`.

    The escape character allows the separator to be used within a substring.

    Args:

        s: The string to be split.

        sep: The separator character.

        escape_char: The escape character.

    """

    in_escape = False

    result = []

    substring = ''

    for c in s:

        if c == escape_char and not in_escape:

            in_escape = True

            continue

        elif c == sep and not in_escape:

            result.append(substring)

            substring = ''

        else:

            substring += c

        in_escape = False

    if substring:

        result.append(substring)

    return result

