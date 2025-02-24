import numpy as np



def impute_mean(data: List[List[Union[float, int, None]]]) -> List[List[float]]:

    """Imputes missing values in a list of lists (data frame) with the mean of existing values in the same column.



    Args:

        data: The input dataset as a list of lists.



    Returns:

        A new dataset with the missing values replaced by the mean of the non-missing values in the same column.

    """

    arr = np.array(data)

    means = np.nanmean(arr, axis=0)

    inds = np.where(np.isnan(arr))

    arr[inds] = np.take(means, inds[1])

    return arr.tolist()

