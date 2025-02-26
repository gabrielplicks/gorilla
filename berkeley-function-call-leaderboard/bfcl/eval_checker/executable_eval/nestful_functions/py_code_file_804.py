from typing import List, Union



def check_target_in_list(string_list: Union[List[str], None], target: Union[str, None]) -> bool:

    """Checks if the target string appears in the list of strings.

    Args:

        string_list: The list of strings to search in.

        target: The target string to search for.

    """

    if string_list is None or target is None:

        return False

    target = target.lower()

    for s in string_list:

        if s.lower() == target:

            return True

    return False

