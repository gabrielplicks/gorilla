import math



def is_product(n: int) -> bool:

    """Checks if an integer `n` is the product of two integers between 2 and `n - 1`.



    Args:

        n: The integer to check.



    Returns:

        True if `n` is the product of two integers between 2 and `n - 1`, False otherwise.

    """

    if n < 4:

        return False



    for i in range(2, int(math.sqrt(n)) + 1):

        if n % i == 0:

            return True

    return False

