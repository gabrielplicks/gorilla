import datetime

from typing import Any



def is_datetime_subclass(obj: Any) -> bool:

    """Checks whether the input object is an instance of the `datetime.datetime` class or a subclass of it.



    Args:

        obj: The input object to be checked.



    Returns:

        True if the input object is an instance of a subclass of `datetime.datetime`, False otherwise.

    """

    return isinstance(obj, datetime.datetime)

