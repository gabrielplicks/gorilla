from typing import Dict, List

import math



def normalize_data(data: Dict[str, List[float]]) -> Dict[str, List[float]]:

    """Normalizes the values of each key in a dictionary.



    Args:

        data: A dictionary where the values are lists of floats.



    Returns:

        A dictionary with the same keys as the input dictionary, where the values are the normalized values.

    """

    normalized_data = {}

    for key, values in data.items():

        mean = sum(values) / len(values)

        std = (sum([(x - mean)**2 for x in values]) / len(values))**0.5

        normalized_data[key] = [(x - mean) / std for x in values]

    return normalized_data

