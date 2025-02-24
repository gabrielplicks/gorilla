from typing import Dict



def update_d1_with_d2(d1: Dict[str, list], d2: Dict[str, int]) -> None:

    """Updates d1 with the keys and values from d2, where the values are lists with the original key's value appended to the end of the list.



    Args:

        d1: The dictionary to be updated.

        d2: The dictionary containing the keys and values to be added to d1.

    """

    for key, value in d2.items():

        if key not in d1:

            d1[key] = [value]

        else:

            d1[key].append(value)

