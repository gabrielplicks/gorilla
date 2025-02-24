import numpy as np



def split_tensor(data: np.ndarray, split_ratio: float) -> tuple:

    """Splits a tensor with dimension (N, D, H) into the training set and validation set.



    Args:

        data: A tensor with dimension (N, D, H).

        split_ratio: A floating point number in the range [0, 1].



    Returns:

        A tuple of two tensors, each with shape (N, D, split_ratio * H).

    """

    N, D, H = data.shape

    split_index = int(round(split_ratio * H))

    train_set = data[:, :, :split_index]

    val_set = data[:, :, split_index:]



    return train_set, val_set

