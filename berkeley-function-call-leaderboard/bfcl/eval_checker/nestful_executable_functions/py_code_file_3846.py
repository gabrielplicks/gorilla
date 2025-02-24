import reprlib

import sys



def repr_with_default_encoding(obj: object) -> str:

    """Re-implementation of Python's `repr` function with `sys.getdefaultencoding()` support.



    Args:

        obj: The object to be represented.



    Returns:

        The representation of the object.

    """

    repr_obj = reprlib.Repr()

    repr_obj.maxstring = sys.maxsize

    repr_obj.maxother = sys.maxsize

    repr_obj.maxlevel = sys.maxsize



    return repr_obj.repr(obj)

