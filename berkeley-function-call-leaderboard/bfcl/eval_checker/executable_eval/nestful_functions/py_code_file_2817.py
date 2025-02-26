import json



def indent_and_pretty_print_json(json_object: dict) -> str:

    """

    Indents and pretty-prints a JSON object.

    Args:

        json_object: The JSON object to be indented and pretty-printed.

    """

    return json.dumps(json_object, indent=4, sort_keys=True)

