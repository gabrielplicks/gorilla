from typing import Dict



def function_1(a: int, b: int, c: int) -> Dict[str, int]:

    """

    Returns a dictionary with the variable names and their corresponding values.



    Args:

        a: The first variable.

        b: The second variable.

        c: The third variable.



    Returns:

        A dictionary with the variable names and their corresponding values.

    """

    return {key: value for key, value in locals().items() if key != 'a'}

