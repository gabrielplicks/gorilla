from typing import Union



def generate_cache_key(string_param: str, int_param: int, bool_param: bool) -> str:

    """Generates a unique key based on the given parameters.



    Args:

        string_param: The string parameter.

        int_param: The integer parameter.

        bool_param: The boolean parameter.



    Returns:

        A unique key based on the given parameters.

    """

    key = f"{string_param}{int_param}"

    if bool_param:

        key += "True"

    else:

        key += "False"

    return key

