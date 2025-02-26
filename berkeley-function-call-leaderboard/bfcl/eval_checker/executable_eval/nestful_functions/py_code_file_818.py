from typing import List



def get_sequences_from_sequence_names(sequence_names: List[str]) -> List[str]:

    """Extracts the sequence names without the sequence index from a list of sequence names.



    Args:

        sequence_names: A list of sequence names in the format `sequence_name#sequence_index`.



    Returns:

        A list of sequence names without the sequence index.

    """

    sequences = [sequence_name.split('#')[0] for sequence_name in sequence_names]

    return sequences

