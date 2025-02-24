from typing import Any



def is_model(obj: Any) -> str:

    """Determines if an object is an instance of a class named `Model`.



    Args:

        obj: The object to check.



    Returns:

        The class name of the object if it is an instance of `Model`.



    Raises:

        AttributeError: If the object is not an instance of `Model`.

    """

    if isinstance(obj, Model):

        return obj.__class__.__name__

    else:

        raise AttributeError("The object is not an instance of Model.")

