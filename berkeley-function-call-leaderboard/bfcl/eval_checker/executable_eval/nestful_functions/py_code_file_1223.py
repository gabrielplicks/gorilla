from typing import List, Dict



def create_unique_item_dictionary(items1: List[str], items2: List[str]) -> Dict[str, List[str]]:

    """Creates a dictionary of unique items from two lists.

    The keys of the dictionary are the values from the first list,

    and the value for each key is a list of items from the second list

    that are different from the corresponding key.

    Args:

        items1: The first list of items.

        items2: The second list of items.

    """

    unique_items = {}

    for index, item in enumerate(items1):

        second_item = items2[index]

        if item not in unique_items:

            unique_items[item] = [second_item]

        else:

            if second_item not in unique_items[item]:

                unique_items[item].append(second_item)

    return unique_items

