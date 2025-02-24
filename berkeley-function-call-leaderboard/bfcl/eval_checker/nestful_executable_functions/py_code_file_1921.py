from typing import Dict



def print_type_hints(hints: Dict[str, type]) -> str:

    """

    Prints type hints for a dictionary of variables and their associated type hints.



    Args:

        hints: A dictionary of variables and their associated type hints.

    """

    lines = [f"{key}: {value.__name__}" for key, value in hints.items()]

    return "\n".join(lines)

