import numpy as np



def format_list_of_ints(arr: np.ndarray) -> str:

    """Formats a list of integers into a string with the following form: "3 unique values are: 1, 2, 3 (counts: 1, 2, 3)".

    Uses Numpy to extract and count the unique values in an array.



    Args:

        arr: The input array of integers.



    Returns:

        A formatted string containing the unique values and their counts.

    """

    unique_values, counts = np.unique(arr, return_counts=True)

    unique_values_str = ", ".join(str(x) for x in unique_values)

    counts_str = ", ".join(str(x) for x in counts)

    return f"{len(unique_values)} unique values are: {unique_values_str} (counts: {counts_str})"

