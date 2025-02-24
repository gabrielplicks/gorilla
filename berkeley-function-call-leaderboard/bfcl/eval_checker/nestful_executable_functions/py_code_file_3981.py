from typing import List



def sequence_handling(sequence: List[int]) -> List[int]:

    """Truncates a sequence of integers to the maximum length of 8 and pads it with 0s to the maximum length.



    Args:

        sequence: The input sequence of integers.



    Returns:

        The truncated and padded sequence.

    """

    truncated_sequence = sequence[:8]  # Truncate to the maximum length of 8

    padded_sequence = truncated_sequence + [0] * (8 - len(truncated_sequence))  # Pad with 0s to the maximum length

    return padded_sequence

