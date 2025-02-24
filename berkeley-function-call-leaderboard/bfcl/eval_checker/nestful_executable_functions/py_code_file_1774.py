from typing import List



def generate_constraints(n: int, var_prefix: str = "var_", index_format: callable = str) -> List[str]:

    """Generates a list of constraints for a boolean variable.

    Args:

        n: A positive integer representing the number of constraints to generate.

        var_prefix: A string representing the prefix for the variable names.

        index_format: A callable representing the format for the variable indices.

    """

    return [f"not {var_prefix}{index_format(i)}" for i in range(1, n + 1)]

