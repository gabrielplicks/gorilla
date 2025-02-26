from typing import Dict, List, Any



def find_index_by_params(lst: List[Dict[str, Any]], params: Dict[str, Any]) -> int:

    """Finds the index of the first element in the list that contains all parameters with matching values.



    Args:

        lst: A list of dictionaries.

        params: A dictionary of parameters.



    Returns:

        The index of the first element in the list that contains all parameters with matching values, or None if no such element is found.

    """

    if not lst:

        return None



    for i, d in enumerate(lst):

        if all(d.get(k) == v for k, v in params.items()):

            return i



    return None

