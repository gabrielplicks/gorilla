from typing import Dict



def average_numeric_values(dictionary: Dict[str, float]) -> float:

    """Calculates the average value of all numeric entries in the input dictionary.



    Args:

        dictionary: The input dictionary.



    Returns:

        The average value of all numeric entries in the dictionary. If the dictionary is empty or does not contain any numeric values, returns 0.

    """

    if not dictionary:

        return 0



    total = 0

    count = 0



    for value in dictionary.values():

        if isinstance(value, (int, float)):

            total += value

            count += 1



    if count == 0:

        return 0



    return total / count

