import json

from typing import List, Tuple



def get_top_3(file_path: str) -> List[Tuple[str, int]]:

    """Reads a JSON-formatted file and returns a list of tuples with the top 3 pairs of keys and values, sorted in descending order of values.



    Args:

        file_path: The path to the JSON-formatted file.



    Returns:

        A list of tuples with the top 3 pairs of keys and values.

    """

    with open(file_path, 'r') as f:

        data = json.load(f)

    top_list = []

    for key, value in sorted(data.items(), key=lambda item: item[1], reverse=True):

        top_list.append((key, value))

        if len(top_list) == 3:

            break

    return top_list

