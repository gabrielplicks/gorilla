from typing import List



def distinct_lines(lines: List[str]) -> List[str]:

    """Returns a list of distinct lines in the original list.

    Args:

        lines: A list of strings.

    """

    if not isinstance(lines, list) or not lines:

        return []

    s = set(lines)

    return sorted(list(s))

