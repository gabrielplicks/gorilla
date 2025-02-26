from typing import Dict, List, Tuple, Union



def convert_filter(dict_input: Dict[Union[str, int], Union[Dict, List]]) -> List[Tuple[Union[str, int], Union[Dict, List]]]:

    """Converts a dictionary of key-value pairs into a list of tuples, `(key, value)`, where the key is a string (or an integer), and the value is a dictionary (or a list). Filters the list by excluding any tuples where the key is `"foo"` and the value is a list.



    Args:

        dict_input: The input dictionary.



    Returns:

        A list of tuples, `(key, value)`, where the key is a string (or an integer), and the value is a dictionary (or a list).

    """

    return [(k, v) for k, v in dict_input.items() if k != "foo" or not isinstance(v, list)]

