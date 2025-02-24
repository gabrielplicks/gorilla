from typing import List



def interleave_sequences(sequence1: List[int], sequence2: List[int]) -> List[int]:

    """Interleaves two sequences of integers.



    Args:

        sequence1: The first sequence of integers.

        sequence2: The second sequence of integers.



    Returns:

        A new sequence formed by interleaving the two given sequences.

    """

    new_sequence = []

    for i in range(max(len(sequence1), len(sequence2))):

        if i < len(sequence1):

            new_sequence.append(sequence1[i])

        if i < len(sequence2):

            new_sequence.append(sequence2[i])

    return new_sequence

