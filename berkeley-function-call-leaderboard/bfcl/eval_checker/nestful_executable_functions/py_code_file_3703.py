from typing import List, Dict



def extract_paths(dicts: List[Dict[str, str]]) -> List[str]:

    """Extracts the `path` field from each dictionary in a list of dictionaries.



    Args:

        dicts: A list of dictionaries.



    Returns:

        A list of strings representing the `path` field from each dictionary.

    """

    paths = []

    for d in dicts:

        if 'path' in d:

            paths.append(d['path'])

    return paths

