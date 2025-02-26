from typing import Any



def get_shape_str(obj: Any) -> str:

    """Returns a string of the form "width x height" for an object with width and height attributes.



    Args:

        obj: The object with width and height attributes.



    Raises:

        TypeError: If the object does not have width and height attributes.

    """

    if not hasattr(obj, 'width') or not hasattr(obj, 'height'):

        raise TypeError('Object must have width and height attributes')

    return f'{obj.width} x {obj.height}'

