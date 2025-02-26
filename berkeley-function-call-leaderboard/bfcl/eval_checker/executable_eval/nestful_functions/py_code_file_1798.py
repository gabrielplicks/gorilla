from typing import Any



def caller_name() -> str:

    """Returns the name of the caller of the function.



    Args:

        None



    Returns:

        The name of the caller function.

    """

    return caller_name.caller



def foo() -> str:

    """Sets the caller name to 'foo' and returns the caller name.



    Args:

        None



    Returns:

        The name of the caller function.

    """

    caller_name.caller = 'foo'

    return caller_name()



def bar() -> str:

    """Sets the caller name to 'bar' and returns the caller name.



    Args:

        None



    Returns:

        The name of the caller function.

    """

    caller_name.caller = 'bar'

    return caller_name()

