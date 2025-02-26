from typing import Dict



def read_expressions(file_path: str) -> Dict[str, float]:

    """Reads a file containing expression-value pairs and returns a dictionary of expression-value pairs.



    Args:

        file_path: The path to the file containing expression-value pairs.



    Returns:

        A dictionary of expression-value pairs.

    """

    expressions = {}

    with open(file_path, "r") as file:

        for line in file:

            expression, value = line.split("=")

            expression = expression.strip()

            value = float(value.strip())

            expressions[expression] = value

    return expressions

