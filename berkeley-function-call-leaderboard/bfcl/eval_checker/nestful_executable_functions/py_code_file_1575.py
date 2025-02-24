from typing import Any



def example_function(first_arg: Any, second_arg: Any, third_arg: Any) -> tuple:

    """

    Returns a tuple containing the values of the three arguments in the reverse order.

    Args:

        first_arg: The first argument.

        second_arg: The second argument.

        third_arg: The third argument.

    """

    return tuple([third_arg, second_arg, first_arg])

