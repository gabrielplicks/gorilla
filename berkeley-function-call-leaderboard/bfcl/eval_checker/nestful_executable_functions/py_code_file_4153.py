from typing import List, Dict



def default_fruit_dict(fruit_list: List[str], default_value: int = 0) -> Dict[str, int]:

    """Creates a dictionary with fruit names as keys and default values.



    Args:

        fruit_list: A list of fruit names.

        default_value: The default value for each fruit name in the dictionary.

    """

    return {fruit: default_value for fruit in fruit_list}

