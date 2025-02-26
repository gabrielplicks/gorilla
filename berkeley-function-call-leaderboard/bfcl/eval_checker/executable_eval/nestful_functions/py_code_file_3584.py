from typing import List



def is_built_in(built_ins: List[str], func: str) -> bool:

    """Checks if a list of Python built-in functions contains a given function name.



    Args:

        built_ins: A list of strings representing Python built-in function names.

        func: A string representing a function name.



    Raises:

        ValueError: If the built-ins list is empty.

    """

    if len(built_ins) < 1:

        raise ValueError('The built-ins list must contain at least one element.')



    filtered_list = list(filter(lambda x: x == func, built_ins))

    return bool(filtered_list)

