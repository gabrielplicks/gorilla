from typing import List



def generate_fibonacci_numbers(n: int) -> List[int]:

    """Generates the first 'n' Fibonacci numbers.



    Args:

        n: A positive integer.



    Raises:

        Exception: If 'n' is not a positive integer.

    """

    if n <= 0:

        raise Exception("The input 'n' must be a positive integer.")



    fibonacci_numbers = [0, 1]



    for i in range(2, n):

        fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2])



    return fibonacci_numbers

