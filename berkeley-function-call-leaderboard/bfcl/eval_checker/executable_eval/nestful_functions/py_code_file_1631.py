import re

from typing import Optional



def transcribe_dna_to_rna(dna_sequence: str) -> Optional[str]:

    """Transcribes a DNA sequence to its corresponding RNA sequence.



    Args:

        dna_sequence: The DNA sequence to transcribe.



    Returns:

        The transcribed RNA sequence, or None if the input DNA sequence is invalid.

    """

    if not re.match(r'^[ACGT]+$', dna_sequence):

        return None

    rna_sequence = ''.join('U' if ch == 'A' else 'C' if ch == 'G' else 'A' if ch == 'T' else 'G' for ch in dna_sequence)



    return rna_sequence

