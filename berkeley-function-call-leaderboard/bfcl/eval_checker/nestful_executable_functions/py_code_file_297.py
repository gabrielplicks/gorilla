import re

from typing import Tuple



def replace_line_with_pattern(text: str, pattern: str, repl: str) -> str:

    """Replaces a line that matches the regular expression `pattern` in the given `text` with the replacement `repl`.



    Args:

        text: The input text.

        pattern: The regular expression pattern to search for.

        repl: The replacement string.



    Returns:

        The modified `text` as a string.

    """

    lines = text.split("\n")

    for i, line in enumerate(lines):

        if re.search(pattern, line):

            lines[i] = re.sub(pattern, repl, line)

            break

    return "\n".join(lines)

