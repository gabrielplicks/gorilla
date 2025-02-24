from typing import Dict, List, Any



def apply_hash(objects: List[Any], k: int) -> Dict[Any, int]:

    """Applies the hash function to each object in the list and stores the object in a dictionary if the hash value is less than `k`.

    If an object is already present in the dictionary, a `ValueError` exception is raised.



    Args:

        objects: The list of objects that will be processed.

        k: The threshold value for the hash function.



    Returns:

        A dictionary that stores the objects for which the hash value is less than `k`.

    """

    dictionary = {}

    for obj in objects:

        hashed_value = hash(obj)

        if hashed_value < k:

            if obj in dictionary:

                raise ValueError('Object already present in the dictionary')

            else:

                dictionary[obj] = hashed_value

        else:

            continue

    return dictionary

