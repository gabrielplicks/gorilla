from typing import List



def count_hashes(text: str) -> int:

    """Counts the number of '#' characters in each row of `text` and returns the sum of the counts.



    Args:

        text: An input string containing rows and columns of characters.



    Returns:

        The sum of the counts for each row.

    """

    rows: List[str] = text.split('\n')

    counts: List[int] = [row.count('#') for row in rows]

    return sum(counts)

