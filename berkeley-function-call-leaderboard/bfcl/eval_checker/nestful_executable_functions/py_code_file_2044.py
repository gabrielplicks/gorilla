from typing import Dict



def modify_args(args: Dict[str, str]) -> Dict[str, str]:

    """Modifies the value of a specific key in a dictionary and returns a new dictionary.

    Args:

        args: The dictionary to modify.

    Returns:

        A new dictionary with the modified key and value.

    """

    new_args = args.copy()  # Copy the original dictionary

    new_args["key"] = "modified_value"  # Modify the value of the specified key

    return new_args  # Return the modified dictionary

