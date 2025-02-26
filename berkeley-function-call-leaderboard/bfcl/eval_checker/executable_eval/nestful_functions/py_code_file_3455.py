import torch

from typing import List



def gen_mask(sequences: List[List[int]]) -> torch.Tensor:

    """Generates a boolean mask that corresponds to the sequences' length.



    Args:

        sequences: A list of sequences, each with a different length.



    Returns:

        A pytorch tensor with the same shape as the input list.

    """

    max_length = max(len(sequence) for sequence in sequences)

    mask = torch.full((len(sequences), max_length), False, dtype=torch.bool)

    for i, sequence in enumerate(sequences):

        mask[i, :len(sequence)] = True

    return mask

