from typing import Dict, Any



def convert_to_map(lst: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:

    """Converts a list of objects into a dictionary, where the keys are the values from an object's `id` attribute and the values are the objects themselves.



    Args:

        lst: The list of objects to convert.



    Returns:

        A dictionary where the keys are the values from an object's `id` attribute and the values are the objects themselves.

    """

    return {obj.get('id', idx): obj for idx, obj in enumerate(lst, 1)}

