import numpy as np



def get_next_batch(data: np.ndarray, labels: np.ndarray, batch_size: int, train: bool = True) -> tuple:

    """

    Returns a batch of data and labels for training or inference.



    Args:

        data: A numpy array of input data.

        labels: The corresponding numpy array of labels.

        batch_size: An integer indicating the size of each batch.

        train: A boolean indicating whether the batch is for training (default) or inference (if set to False).

    """

    permutation = np.random.permutation(data.shape[0])

    data = data[permutation]

    labels = labels[permutation]



    if train:

        return data[:batch_size], labels[:batch_size]

    else:

        return data, labels

