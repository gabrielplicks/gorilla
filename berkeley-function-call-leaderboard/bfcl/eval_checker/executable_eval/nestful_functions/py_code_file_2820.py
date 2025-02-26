from typing import List



def replace_with_target(src_list: List[int], target: int, replacement: int) -> List[int]:

    """Replaces every occurrence of `target` in `src_list` with `replacement`.

    If `target` does not occur in `src_list`, returns an empty list.

    Args:

        src_list: A list of integers.

        target: The integer to be replaced.

        replacement: The integer to replace `target` with.

    """

    if target not in src_list:

        return []

    return [replacement if element == target else element for element in src_list]

