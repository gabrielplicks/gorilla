from typing import List



def generate_increasing_sequences(n: int, k: int) -> List[List[int]]:

    """Generates a list of all increasing sequences of length k that are composed of the first n natural numbers.



    Args:

        n: The number of natural numbers to use.

        k: The length of the sequences.



    Returns:

        A list of increasing sequences of length k that are composed of the first n natural numbers.

    """

    if k == 1:

        return [[i] for i in range(n, 0, -1)]



    sequences = []

    for i in range(n, 0, -1):

        sequences_without_i = generate_increasing_sequences(i - 1, k - 1)

        for sequence in sequences_without_i:

            sequences.append([i] + sequence)



    return sequences

