from typing import List, Dict



def generate_zero_values(input_list: List[Dict[str, int]]) -> List[Dict[str, int]]:

    """Generates a new list of dictionaries with the same coordinates as the input list, but with the value set to 0.



    Args:

        input_list: A list of dictionaries, where each dictionary represents a set of coordinates with a corresponding value.



    Returns:

        A new list of dictionaries with the same coordinates as the input list, but with the value set to 0.

    """

    return [{**entry, "value": 0} for entry in input_list]

