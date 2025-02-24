from typing import Dict, List, Union



def transfer_backward(forward_dict: Dict[int, str], value: str) -> Union[List[int], None]:

    """

    Transfers a value from a forward dictionary to a backward dictionary.



    Args:

        forward_dict: A dictionary that maps keys to values.

        value: The value to be looked up in the dictionary.



    Returns:

        A list of keys that map to the given value in the forward dictionary, or None if the forward dictionary is empty.

    """

    if not forward_dict:

        return None

    backward_dict = {v: [k for k in forward_dict if forward_dict[k] == v] for v in set(forward_dict.values())}

    return backward_dict.get(value, [])

