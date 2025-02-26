from collections import OrderedDict

from typing import Any, Dict, TypeVar



K = TypeVar("K")

V = TypeVar("V")



def get_last_key(ordered_dict: Dict[K, V]) -> K:

    """Returns the key of the last element in an OrderedDict.



    Args:

        ordered_dict: The OrderedDict object.



    Returns:

        The key of the last element in the OrderedDict.

    """

    for key, _ in ordered_dict.items():

        pass

    return key

