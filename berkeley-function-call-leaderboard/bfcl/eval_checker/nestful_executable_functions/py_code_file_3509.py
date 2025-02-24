from typing import Tuple



def product_string(n: int, m: int) -> str:

    """Generates a formatted string representing the product of two positive integers.

    The string has `n` rows of `m` asterisks each, where each row is separated by a newline character.

    Args:

        n: The number of rows.

        m: The number of asterisks in each row.

    """

    row = "*" * m  # Generate one row of asterisks

    rows = "\n".join([row] * n)  # Join rows with newline characters

    return rows

