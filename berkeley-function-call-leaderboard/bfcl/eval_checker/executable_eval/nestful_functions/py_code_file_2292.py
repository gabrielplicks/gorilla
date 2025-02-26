from typing import List



def generate_fibonacci_like(a: int, b: int, n: int) -> List[int]:

    """Generates a Fibonacci-like sequence of numbers.



    Args:

        a: The first number in the sequence.

        b: The second number in the sequence.

        n: The number of elements to generate.



    Returns:

        A list of `n` numbers in the sequence.

    """

    sequence = []

    if not (isinstance(a, int) and isinstance(b, int)):

        raise ValueError("a and b must be integers")

    if not isinstance(n, int) or n <= 0:

        raise ValueError("n must be a positive integer")

    sequence.append(a)

    sequence.append(b)

    for i in range(2, n):

        sequence.append(sequence[i-1] + sequence[i-2])

    return sequence

