from typing import Dict, Any, Type



def match_schema(input_dict: Dict[str, Any], type_annotations: Dict[str, Any]) -> bool:

    """Checks if the input dictionary matches the given type annotations.



    Args:

        input_dict: The input dictionary to check.

        type_annotations: The type annotations to match against.



    Returns:

        True if the input dictionary matches the type annotations, False otherwise.

    """

    if not isinstance(input_dict, dict):

        return False



    for k, v in type_annotations.items():

        if k not in input_dict:

            return False



        if isinstance(v, dict):

            if not match_schema(input_dict[k], v):

                return False



        elif isinstance(v, Type):

            if not isinstance(input_dict[k], v):

                return False



    return True

