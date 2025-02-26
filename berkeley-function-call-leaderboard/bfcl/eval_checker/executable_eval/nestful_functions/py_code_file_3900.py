from typing import Dict, List



def relative_volume(dna_sequences: List[Dict[str, float]]) -> Dict[str, float]:

    """Calculates the relative volume of each DNA sequence when compared to the total volume of all DNA sequences.



    Args:

        dna_sequences: A list of DNA sequences, where each sequence is a dictionary with keys `id` (sequence ID) and `volume` (sequence volume).



    Returns:

        A dictionary where each key corresponds to a DNA sequence ID and its value is the relative volume as a percentage.

    """

    total_volume = sum(sequence['volume'] for sequence in dna_sequences)

    relative_volumes = {}

    for sequence in dna_sequences:

        relative_volume = (sequence['volume'] / total_volume) * 100

        relative_volumes[sequence['id']] = relative_volume



    return relative_volumes

