import copy



def create_deep_copy(obj: object) -> object:

    """Creates a deep copy of the given object's state.



    Args:

        obj: The object to create a deep copy of.



    Returns:

        A deep copy of the given object.

    """

    return copy.deepcopy(obj)

