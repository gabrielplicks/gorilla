from typing import List



def extract_names_with_underscore(strings: List[str]) -> List[str]:

    """Extracts strings containing underscores from a list of strings.



    Args:

        strings: A list of strings.



    Returns:

        A new list containing only the strings that contain underscores.

    """

    return [string for string in strings if '_' in string]

