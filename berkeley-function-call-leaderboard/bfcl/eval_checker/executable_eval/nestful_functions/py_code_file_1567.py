def euclid_gcd(a: int, b: int) -> int:

    """Calculates the GCD of two integers using Euclid's algorithm.

    Args:

        a: The first integer.

        b: The second integer.

    """

    if b == 0:

        return a

    else:

        return euclid_gcd(b, a % b)

