from typing import Any



class Human:

    pass



def is_human_object(x: Any) -> bool:

    """Checks whether the object `x` is an instance of the `Human` class.



    Args:

        x: The object to be checked.



    Returns:

        True if the object `x` is an instance of the `Human` class, False otherwise.

    """

    return isinstance(x, Human)

