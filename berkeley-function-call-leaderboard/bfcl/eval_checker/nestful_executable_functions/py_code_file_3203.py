from typing import List, Set



def filter_data_objects(data_objects: List[object], keys: List[str]) -> List[object]:

    """Filters the data objects based on the provided keys.



    Args:

        data_objects: A list of data objects.

        keys: A list of keys.



    Returns:

        A list containing the data objects associated with the provided keys.

    """

    if len(keys) == 0:

        return []

    return [data_object for data_object in data_objects if data_object.key in set(keys)]

