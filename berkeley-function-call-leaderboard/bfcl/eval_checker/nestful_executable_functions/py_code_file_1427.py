from typing import List



def count_gaps(seq1: List[str], seq2: List[str]) -> int:

    """Counts the number of gaps in an alignment between two DNA sequences.



    A gap is represented by a `-` in one or both sequences.



    Args:

        seq1: The first DNA sequence.

        seq2: The second DNA sequence.



    Raises:

        ValueError: If the two sequences are not the same length.

    """

    if len(seq1) != len(seq2):

        raise ValueError("The two sequences must be the same length")



    num_gaps = 0

    for i in range(len(seq1)):

        if seq1[i] == '-' or seq2[i] == '-':

            num_gaps += 1

    return num_gaps

