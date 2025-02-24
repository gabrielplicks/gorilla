from typing import Union



def generate_target(content: str, n: int) -> str:

    """Generates a target string consisting of `n` repetitions of `content` delimited by `'\n'`.

    The first `n - 1` repetitions are prefixed with `'> '` and the last one is prefixed with `'>> '` if `n > 1`.

    Args:

        content: The string to be repeated.

        n: The number of repetitions.

    """

    if n == 1:

        return content

    elif n > 1:

        target = content + '\n' + '> ' + content

        return target * (n - 2) + '\n>> ' + content

    else:

        return ''

