from typing import List, Union



def calculate_stack_name(stack_path: str) -> str:

    """Calculates the name of the stack from the given stack path.



    Args:

        stack_path: The path to a CloudFormation stack, which is in the format of

            'stack_project_name/stack_name' or just 'stack_name'.



    Raises:

        ValueError: If the argument does not follow the expected format.

    """

    parts: List[str] = stack_path.split('/')

    if len(parts) == 1:

        stack_name: str = parts[0]

    elif len(parts) == 2:

        stack_name: str = parts[1]

    else:

        raise ValueError("Invalid stack path: expected format is 'stack_project_name/stack_name'")

    return stack_name

