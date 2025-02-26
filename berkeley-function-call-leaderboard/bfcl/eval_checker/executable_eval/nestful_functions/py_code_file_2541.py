from typing import Dict, Any



def complement_dna_sequence(dna_sequence: str) -> str:

    """Complements a DNA sequence by mapping nucleotides according to the given rules.



    Args:

        dna_sequence: The DNA sequence to be complemented.



    Returns:

        The complemented DNA sequence.

    """

    complement_map: Dict[str, str] = {

        'A': 'T',

        'C': 'G',

        'G': 'C',

        'T': 'A'

    }



    complemented_sequence: str = ''



    for nucleotide in dna_sequence:

        complemented_sequence += complement_map.get(nucleotide, nucleotide)



    return complemented_sequence

