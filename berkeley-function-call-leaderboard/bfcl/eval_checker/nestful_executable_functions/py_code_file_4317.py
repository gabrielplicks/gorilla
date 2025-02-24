import json

import statistics

from typing import List



def compute_statistic(file_path: str, attribute: str, statistic: str) -> float:

    """Computes a statistic (e.g., average, median, or other measure) on the object's attributes.



    Args:

        file_path: The path to the JSON file.

        attribute: The attribute to compute the statistic on.

        statistic: The statistic to compute (e.g., 'mean', 'median', 'mode', etc.).



    Returns:

        The computed statistic.

    """

    with open(file_path, 'r') as f:

        obj: List[dict] = json.load(f)

    values = [item[attribute] for item in obj]

    return getattr(statistics, statistic)(values)

