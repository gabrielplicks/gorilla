from typing import Dict, Any



def left_outer_join(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:

    """Performs a left-outer join operation on two dictionaries based on common keys.



    Args:

        left: The first dictionary.

        right: The second dictionary.



    Returns:

        A new dictionary with the joined result.

    """

    result = {}



    for key in left:

        if key in right:

            result[key] = (left[key], right[key])

        else:

            result[key] = (left[key], None)



    for key in right:

        if key not in left:

            result[key] = (None, right[key])



    return result

