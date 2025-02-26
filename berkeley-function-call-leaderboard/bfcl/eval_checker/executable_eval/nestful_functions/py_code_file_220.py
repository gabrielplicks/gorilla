from typing import Union



def fizz_buzz(n: Union[int, float]) -> str:

    """

    Returns a string based on the truthiness and divisibility of a number.



    Args:

        n: The number to check.

    """

    if n:  # Check if number is truthy

        if n % 3 == 0 and n % 5 == 0:

            return "FizzBuzz"

        elif n % 3 == 0:

            return "Fizz"

        elif n % 5 == 0:

            return "Buzz"

        else:

            return "1"

    else:

        return ""  # Return empty string if number is falsy

