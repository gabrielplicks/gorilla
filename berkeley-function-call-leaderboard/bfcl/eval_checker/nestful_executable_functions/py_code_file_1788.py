from typing import Dict



def reverse_complement_dna(dna_sequence: str) -> str:

    """Calculates the reverse complement of a DNA sequence.



    Args:

        dna_sequence: The DNA sequence to calculate the reverse complement of.



    Returns:

        The reverse complement of the input DNA sequence.

    """

    complement_map: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

    reverse_complement: str = ""

    for nucleotide in dna_sequence[::-1]:

        reverse_complement += complement_map[nucleotide]

    return reverse_complement

