from typing import Dict



def translate_values(values: Dict[str, any]) -> any:

    """Translates the value in `"from"` to the corresponding value in `"to"`.

    Args:

        values: A dictionary with two possible keys: `"from"` and `"to"`.

    """

    for key in values.keys():

        if key == "from" and values[key] in values["to"]:

            return values["to"][values[key]]

    return None

