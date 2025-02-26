from typing import Union



def generate_drs_uri(obj_id: Union[str, None]) -> str:

    """Generates a DRS URI for accessing an object on the DRS server.

    Args:

        obj_id: The object ID.

    """

    if not obj_id:

        raise ValueError("Object ID cannot be empty.")

    if not obj_id.startswith("/"):

        obj_id = "/" + obj_id

    return f"drs://server{obj_id}"

