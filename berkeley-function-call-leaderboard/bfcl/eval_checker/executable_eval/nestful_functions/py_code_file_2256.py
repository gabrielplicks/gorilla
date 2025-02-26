from typing import List, Tuple



def select_odd_values(triples: List[Tuple[int, int, bool]]) -> List[int]:

    """Returns a list of the values at the indices where is_odd is True.



    Args:

        triples: A list of triples of the form (index, value, is_odd), where index and value are integers and is_odd is a boolean.



    Returns:

        A list of the values at the indices where is_odd is True.

    """

    return [value for index, value, is_odd in triples if is_odd]

