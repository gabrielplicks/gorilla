import numpy as np

from typing import List



def batch_process_sequences(sequences: List[List[float]], batch_size: int) -> List[np.ndarray]:

    """Processes the sequences in batches, padding each batch to the length of the longest sequence in the batch.



    Args:

        sequences: A list of sequences.

        batch_size: The number of sequences to process in each batch.



    Returns:

        A list of padded batches.

    """

    batches = []

    for i in range(0, len(sequences), batch_size):

        padded_batch = np.zeros((batch_size, max(len(seq) for seq in sequences[i:i+batch_size])))

        for j, seq in enumerate(sequences[i:i+batch_size]):

            padded_batch[j, :len(seq)] = seq

        batches.append(padded_batch)

    return batches

