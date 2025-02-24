from typing import Dict



def num_distinct_values(d: Dict[int, str]) -> int:

    """Calculates the number of distinct values in a dictionary `d`.



    Args:

        d: The dictionary to count distinct values in.



    Returns:

        The number of distinct values in `d`.

    """

    if not d:

        return 0

    return len(set(d.values()))

