from typing import Any, Type



def hasattrs(*args: Any, **kwargs: Any) -> bool:

    """

    Checks if all positional arguments are instances of the class specified in the keyword argument named `class`.

    Returns `True` if all positional arguments are instances of that class, otherwise returns `False`.

    Args:

        args: The positional arguments to check.

        kwargs: The keyword arguments. The keyword argument `class` is required.

    """

    cls: Type[Any] = kwargs.get('class')



    if not cls:

        raise ValueError("The keyword argument 'class' is required.")



    for arg in args:

        if not isinstance(arg, cls):

            return False



    return True

