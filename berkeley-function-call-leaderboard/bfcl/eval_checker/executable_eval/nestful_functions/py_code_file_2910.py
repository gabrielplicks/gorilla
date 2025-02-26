from typing import List



def set_boolean(strings: List[str]) -> List[bool]:

    """Assigns a boolean value to each string based on its value.



    Args:

        strings: A list of strings to assign boolean values to.



    Returns:

        A list of boolean values corresponding to the input strings.

    """

    bool_map = {

        'true': True,

        '1': True,

        'yes': True,

        't': True,

        'y': True,

        'false': False,

        '0': False,

        'no': False,

        'f': False,

        'n': False,

    }

    bool_list = []

    for s in strings:

        s_lower = s.lower()

        bool_value = bool_map.get(s_lower, False)

        bool_list.append(bool_value)

    return bool_list

