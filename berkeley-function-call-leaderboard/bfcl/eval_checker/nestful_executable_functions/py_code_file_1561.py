from typing import List, Union



def pad_sequence(seq: List[Union[int, float]], seq_len: int, pad_token: Union[int, float] = 0) -> List[Union[int, float]]:

    """Pads a sequence to a specified length with a padding token.



    Args:

        seq: The sequence to pad.

        seq_len: The length to pad the sequence to.

        pad_token: The token to use for padding.

    """

    seq_len = max(seq_len, 0)  # Ensure the sequence length is non-negative

    seq = seq[:seq_len]  # Trim the sequence to the specified length

    return seq + [pad_token] * (seq_len - len(seq))  # Pad the sequence with the padding token

