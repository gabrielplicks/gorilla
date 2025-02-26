from typing import List



def join_variables(variables: List[str]) -> str:

    """Generates a string from a list of variable names, with each variable name separated by a period.



    Args:

        variables: A list of variable names.

    """

    return '.'.join(variables)

