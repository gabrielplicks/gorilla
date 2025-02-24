from typing import List, Dict



def select_and_sort(data: List[Dict[str, int]], v: int) -> List[Dict[str, int]]:

    """Selects and sorts dictionaries from a list based on a given value.



    Args:

        data: A list of dictionaries. Each dictionary has a key `k` and a value `v`.

        v: The value to match the `k` key in the dictionaries.



    Returns:

        A sorted list of dictionaries that have a `k` key equal to the given `v` value.

    """

    selected_dicts = []

    for d in data:

        if d['k'] == v:

            selected_dicts.append(d)



    return list(reversed(sorted(selected_dicts, key=lambda d: d['v'])))

