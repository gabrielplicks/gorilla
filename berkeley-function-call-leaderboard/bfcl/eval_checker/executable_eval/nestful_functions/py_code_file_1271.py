from typing import List

def is_string_value(variable_name: str) -> bool:

    """Checks if the value of the given variable is a string.



    Args:

        variable_name: The name of the variable to check.



    Returns:

        True if the value of the variable is a string, False otherwise.

    """

    return isinstance(eval(variable_name), str)

def get_string_value_variables(names: List[str]) -> List[str]:

    """Returns a list of variable names whose values are strings.



    Args:

        names: A list of variable names.



    Returns:

        A list of variable names whose values are strings.

    """

    string_value_variables = []



    for variable_name in names:

        if is_string_value(variable_name):

            string_value_variables.append(variable_name)



    return string_value_variables

