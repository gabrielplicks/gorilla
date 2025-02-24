from typing import Dict, Any



def make_result_dict(id: int, feature: Dict[str, Any]) -> Dict[str, Any]:

    """Creates a dictionary containing the given ID and the feature's properties.

    If either the ID or the feature object is empty, they are not set as keys in the returned dictionary.

    Args:

        id: The ID to be included in the result dictionary.

        feature: The feature object whose properties are to be included in the result dictionary.

    """

    if id is None or feature is None:

        return {'id': id if id is not None else feature}

    result = {}

    if id is not None:

        result['id'] = id

    for key, value in feature.items():

        if value is not None:

            result[key] = value

    return result

