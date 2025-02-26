from typing import Any



def my_and(x: Any, y: Any) -> bool:

    """Emulates the logical AND operation without using the `and` keyword.

    Args:

        x: The first operand.

        y: The second operand.

    """

    if x and y:

        return True

    else:

        return False

