from typing import Union



def nested_to_string(data: Union[list, tuple]) -> list:

    """Converts a nested list or tuple to a list of strings.

    Args:

        data: The input data.

    """

    if isinstance(data, (list, tuple)):

        result = []

        for item in data:

            result.extend(nested_to_string(item))

        return result

    else:

        return [str(data)]

