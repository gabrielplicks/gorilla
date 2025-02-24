import json



def python_to_json(obj: object) -> str:

    """Convert a Python object to a JSON string.



    Args:

        obj: A Python object to convert.

    """

    try:

        return json.dumps(obj)

    except Exception as e:

        raise TypeError(f"Could not convert object to JSON: {e}")

