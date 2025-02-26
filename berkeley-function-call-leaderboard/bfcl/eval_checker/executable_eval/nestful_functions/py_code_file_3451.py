from typing import List



def kmers(dna_string: str, k: int) -> List[str]:

    """Generates all possible k-mers of a given length k from a DNA string.



    Args:

        dna_string: The DNA string to generate k-mers from.

        k: The length of the k-mers to generate.



    Returns:

        A list of k-mers.

    """

    k_mers = []

    for i in range(len(dna_string) - k + 1):

        k_mers.append(dna_string[i:i+k])

    return k_mers

