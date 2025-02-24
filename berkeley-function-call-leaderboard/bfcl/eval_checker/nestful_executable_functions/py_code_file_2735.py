from typing import Any, Dict, List



def group_by_type(values: List[Any]) -> Dict[str, List[Any]]:

    """Groups a list of values by their types.



    Args:

        values: A list of values to be grouped by type.



    Returns:

        A dictionary with the keys as the types of the values and the values as the values grouped by type.

    """

    grouped_values = {

        type(value).__name__: [] for value in values

    }



    for value in values:

        grouped_values[type(value).__name__].append(value)



    return grouped_values

