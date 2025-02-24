from typing import List



def sequences_match(seq1: List[str], seq2: List[str]) -> bool:

    """Determines whether two DNA sequences match.



    Args:

        seq1: The first DNA sequence.

        seq2: The second DNA sequence.



    Returns:

        True if the sequences match, False otherwise.

    """

    if len(seq1) != len(seq2):

        return False



    for i in range(len(seq1)):

        if seq1[i] != seq2[i]:

            return False



    return True

