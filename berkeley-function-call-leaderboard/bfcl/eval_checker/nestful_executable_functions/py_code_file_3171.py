from typing import List, Dict



def combine_data(list1: List[Dict[str, str]], list2: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """Combines two lists of data objects with matching keys.



    Args:

        list1: A list of data objects with `key1`, `key2`, and `key3` keys.

        list2: A list of data objects with `key2`, `key3`, and `key4` keys.



    Returns:

        A list of unique data objects with no duplicate keys.

    """

    combined_data = {

        obj1["key1"]: {**obj1, **obj2}

        for obj1 in list1

        for obj2 in list2

        if obj1["key2"] == obj2["key2"] and obj1["key3"] == obj2["key3"]

    }

    return list(combined_data.values())

