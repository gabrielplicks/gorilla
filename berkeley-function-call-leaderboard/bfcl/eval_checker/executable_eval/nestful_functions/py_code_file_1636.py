from typing import List



def join_with_sign(numbers: List[int], sign: str) -> str:

    """

    Joins a list of numbers with the specified sign.



    Args:

        numbers: A list of integers to be joined.

        sign: A single character string to be used as a separator.



    Raises:

        ValueError: If the sign is not a single character string.

    """

    if not isinstance(sign, str) or len(sign) != 1:

        raise ValueError("Sign must be a single character string.")



    return f"{sign}".join(str(n) for n in numbers)

