from typing import List



def mode_size(mode: str, data: List[int]) -> int:

    """Computes the size of a data set given its mode.



    Args:

        mode: A string representing the data set's mode, "nominal", "ordinal", or "interval".

        data: A list of integers representing the data set.



    Raises:

        ValueError: If the mode is none of "nominal", "ordinal", or "interval".

    """

    if mode == "nominal":

        return len(data)

    elif mode == "ordinal":

        return len(data) + 1

    elif mode == "interval":

        return 2 * len(data) + 1

    else:

        raise ValueError("Invalid mode.")

