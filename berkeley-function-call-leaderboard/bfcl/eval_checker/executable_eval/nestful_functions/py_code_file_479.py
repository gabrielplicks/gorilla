from typing import List, Dict, Tuple



def find_id(target_list: List[Dict], id: int) -> List[Tuple]:

    """Returns a list of matches containing the input ID from the given list of dictionaries.



    Args:

        target_list: The list of dictionaries to search through.

        id: The ID to search for.

    """

    matches = []

    for dictionary in target_list:

        if 'id' in dictionary:

            if dictionary['id'] == id:

                matches.append((id,))

    return matches

