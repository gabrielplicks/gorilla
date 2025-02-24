from typing import List, Dict



def join_version(objects: List[Dict[str, str]]) -> List[str]:

    """Joins the `name` and `version` fields of a list of objects with a pipe (|) character as the separator.



    Args:

        objects: A list of objects, where each object is a dictionary with a `name` field and an optional `version` field.



    Returns:

        A list of strings with the joined `name` and `version` fields.

    """

    versions = {obj["name"]: obj["version"] for obj in objects if "version" in obj}

    return [f"{obj['name']}|{versions.get(obj['name'], '')}" for obj in objects]

