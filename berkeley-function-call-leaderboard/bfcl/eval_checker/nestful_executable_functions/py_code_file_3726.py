from typing import Dict



def calculate_missing_ratios(data: Dict[str, Dict[str, float]]) -> Dict[str, float]:

    """Calculates the ratio of missing values for each key in a dictionary of dictionaries.



    Args:

        data: A dictionary of dictionaries. The inner dictionaries have some missing values.



    Returns:

        A dictionary of ratios indicating how much of the data is present for each key.

        The ratio is a value between 0 and 1, with 1 representing all data and 0 representing no data.

    """

    ratios = {}

    for key, inner_dict in data.items():

        num_values = sum(1 for value in inner_dict.values() if value is not None)

        num_keys = len(inner_dict)

        ratio = num_values / num_keys

        ratios[key] = ratio

    return ratios

