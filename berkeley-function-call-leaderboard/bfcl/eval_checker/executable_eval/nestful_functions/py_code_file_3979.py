from typing import List, Union



def flatten_array(arr: List[Union[List, str, int]]) -> List[Union[str, int]]:

    """Flattens a 2D array recursively and formats string values using the `f''` string formatting syntax.



    Args:

        arr: The 2D array to be flattened.



    Returns:

        The flattened 1D array.

    """

    if not arr:

        return None

    flattened_arr = []

    for elem in arr:

        if isinstance(elem, list):

            flattened_arr.extend(flatten_array(elem))

        elif isinstance(elem, str):

            flattened_arr.append(f'I am a string: {elem}')

        else:

            flattened_arr.append(elem)



    return flattened_arr

