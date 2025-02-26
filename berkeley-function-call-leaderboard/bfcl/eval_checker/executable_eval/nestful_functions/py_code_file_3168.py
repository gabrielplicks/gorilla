from typing import List



def get_modulus_sequence(sequence: List[int], divisor: int) -> List[int]:

    """

    Returns a new sequence where each element is the modulus of its original value and the divisor.



    Args:

        sequence: A list of integers.

        divisor: An integer.



    Returns:

        A list of integers.

    """

    result = []

    for n in sequence:

        result.append(n % divisor)

    return result

