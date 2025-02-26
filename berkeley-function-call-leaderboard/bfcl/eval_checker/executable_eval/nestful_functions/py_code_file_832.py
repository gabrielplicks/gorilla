from typing import Dict



def get_algorithm_name(family_name: str, algorithms: Dict[str, str]) -> str:

    """Returns the algorithm name associated with the given family name.



    Args:

        family_name: The name of the algorithm family.

        algorithms: A dictionary with algorithm family names as keys and algorithm names as values.

    """

    for key, value in algorithms.items():

        if key == family_name:

            return value

