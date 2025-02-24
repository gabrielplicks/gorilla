from typing import Union



def and_op(input1: Union[bool, int], input2: Union[bool, int]) -> bool:

    """Performs the logical AND operation on two boolean inputs.

    Args:

        input1: The first boolean input.

        input2: The second boolean input.

    """

    if input1 and input2:  # Check if both inputs are True

        return True

    else:

        return False

