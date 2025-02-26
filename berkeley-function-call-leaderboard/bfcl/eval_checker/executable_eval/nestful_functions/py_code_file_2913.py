from typing import List, Tuple, Any



def partition_list_by_attribute_value(objects: List[Any], attribute: str, value: Any) -> Tuple[List[Any], List[Any]]:

    """Partitions a list of objects based on a specific attribute value.



    Args:

        objects: A list of objects to partition.

        attribute: The name of the attribute to use for partitioning.

        value: The value to use for partitioning.



    Returns:

        A tuple containing two lists. The first list contains all objects with the attribute value equal to the given value. The second list contains all objects with the attribute value not equal to the given value.

    """

    sublist1 = []

    sublist2 = []



    for obj in objects:

        if getattr(obj, attribute) == value:

            sublist1.append(obj)

        else:

            sublist2.append(obj)



    return sublist1, sublist2

