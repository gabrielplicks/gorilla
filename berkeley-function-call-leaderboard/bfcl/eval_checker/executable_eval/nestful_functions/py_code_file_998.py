from typing import List



def same_group_match(group_ids_1: List[int], group_ids_2: List[int]) -> int:

    """Calculates the number of groups that have at least one member in common between two lists.



    Args:

        group_ids_1: The IDs of the first group.

        group_ids_2: The IDs of the second group.



    Raises:

        Exception: If an error occurs during the calculation.

    """

    try:

        set_1 = set(group_ids_1)

        set_2 = set(group_ids_2)

        return len(set_1.intersection(set_2))

    except Exception as e:

        raise e

