from typing import Any, Type



def generate_doc_string(obj: Any) -> str:

    """Generates a doc string for a given object, including its name and type.



    Args:

        obj: The object to generate a doc string for.



    Returns:

        The generated doc string.

    """

    if not obj.__doc__:

        return f"No doc string found for {obj.__name__}"



    obj_name = obj.__name__

    obj_type = obj.__class__.__qualname__ if hasattr(obj, "__qualname__") else obj.__class__.__name__

    obj_doc_string = obj.__doc__



    return f"{obj_name} ({obj_type}): {obj_doc_string}"

