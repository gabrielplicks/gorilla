from typing import Dict, Any, List



def get_leaf_values(data: Dict[str, Dict[str, Any]]) -> List[Any]:

    """Returns a list of all the values in the leaves of the dictionary.



    Args:

        data: A dictionary of the format {'key1': {'key2': 'value1'}, 'key3': {'key4': 'value2'}}.



    Returns:

        A list of all the values in the leaves of the dictionary.

    """

    result = []



    def traverse(curr, path):

        if isinstance(curr, dict):

            for k, v in curr.items():

                traverse(v, path + [k])

        else:

            result.append(curr)



    traverse(data, [])

    return result

