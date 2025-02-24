from typing import List



def generate_bit_patterns(n: int) -> List[str]:

    """Generates all possible bit patterns of length `n`.



    Args:

        n: The length of the bit patterns.



    Returns:

        A list of all possible bit patterns, where each pattern is represented as a string consisting of '0' and '1' characters.

    """

    if n == 0:

        return ['']

    patterns = []

    for bit in ['0', '1']:

        patterns.extend([bit + p for p in generate_bit_patterns(n - 1)])

    return patterns

