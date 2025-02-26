from typing import List, Dict



def get_dict_from_list(lst: List[str]) -> Dict[str, int]:

    """

    Generates a dictionary with the strings from the input list as keys and their frequencies as values.

    The function is recursive and generates sub-dictionaries for each element in the list.

    Args:

        lst: A list of strings.

    Returns:

        A dictionary with the strings as keys and their frequencies as values.

    """

    if not lst:

        return {}

    first, *rest = lst

    sub_dict = get_dict_from_list(rest)

    sub_dict[f"{first}"] = sub_dict.get(f"{first}", 0) + 1

    return sub_dict

