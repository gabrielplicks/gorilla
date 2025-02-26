from typing import List



def generate_strings(strings: List[str]) -> List[str]:

    """Generates a list of strings in the format `prefix_string_suffix`.



    The return list's strings should be sorted in alphabetical order by the "string" element.

    The function returns an empty list when given an empty list.



    Args:

        strings: A list of strings.



    Returns:

        A list of strings in the format `prefix_string_suffix`.

    """

    if not strings:

        return []

    sorted_strings = sorted(strings, key=lambda x: x)

    output = []

    for string in sorted_strings:

        output.append(f"prefix_{string}_suffix")



    return output

