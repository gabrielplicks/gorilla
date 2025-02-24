from typing import Dict



def count_keys(data: Dict[str, int]) -> Dict[str, int]:

    """Counts the number of keys for each unique value in a dictionary.



    Args:

        data: A dictionary where the keys are strings and the values are integers.



    Returns:

        A dictionary where the keys are the unique values from the input dictionary and the values are the counts of the keys for each unique value.

    """

    counts = {}

    for key, value in data.items():

        if value in counts:

            counts[value] += 1

        else:

            counts[value] = 1

    return counts

