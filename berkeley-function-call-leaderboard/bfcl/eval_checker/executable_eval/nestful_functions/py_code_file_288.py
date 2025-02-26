from typing import Dict, List, Union



def calculate_frequency(elements: List[Union[int, str]]) -> Dict[Union[int, str], int]:

    """Calculates the frequency of each element in a list.



    Args:

        elements: A list of integers or strings.



    Returns:

        A dictionary with the frequency of each element.

    """

    if not elements:

        return {}



    frequency = {}

    for element in elements:

        if element in frequency:

            frequency[element] += 1

        else:

            frequency[element] = 1



    return frequency

