def is_palindrome(x: int) -> bool:

    """

    Checks if an integer is a palindrome.



    Args:

        x: The integer to check.



    Returns:

        True if the integer is a palindrome, False otherwise.

    """

    return reverse_integer(x) == x



def reverse_integer(x: int) -> int:

    """

    Reverses an integer.



    Args:

        x: The integer to reverse.



    Returns:

        The reversed integer.

    """

    sign = 1 if x >= 0 else -1

    x *= sign



    reversed_int = 0

    while x > 0:

        last_digit = x % 10

        reversed_int = reversed_int * 10 + last_digit

        x //= 10



    return reversed_int * sign

