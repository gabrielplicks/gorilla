import torch



def collate(sequences: List[List[int]]) -> torch.Tensor:

    """

    Pads a batch of variable-length sequences to the maximum sequence length in the batch.



    Args:

        sequences: A list of variable-length sequences.



    Returns:

        A padded batch with time steps in the first dimension.

    """

    max_length = max(map(len, sequences))

    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences.permute(1, 0)

