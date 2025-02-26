from typing import List



def neg_span(s: str) -> int:

    """Calculates the longest negation span in a binary string.



    Args:

        s: The binary string.



    Returns:

        The length of the longest negation span.

    """

    longest_span = 0

    current_span = 0

    for c in s:

        if c == "0":

            current_span = 0

        elif c == "1":

            current_span += 1

        longest_span = max(longest_span, current_span)

    return longest_span

