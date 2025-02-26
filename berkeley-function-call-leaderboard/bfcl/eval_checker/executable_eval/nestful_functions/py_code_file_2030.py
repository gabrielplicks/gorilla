from typing import List



def normalize_data_list(data_list: List[float], new_min: float, new_max: float) -> List[float]:

    """Normalizes a list of data points to a new range.



    Args:

        data_list: The list of data points to normalize.

        new_min: The minimum value of the new range.

        new_max: The maximum value of the new range.



    Returns:

        The normalized list of data points.

    """

    original_min = min(data_list)

    original_max = max(data_list)

    new_range = new_max - new_min

    normalized_list = [(x - original_min) / (original_max - original_min) * new_range + new_min for x in data_list]

    return normalized_list

