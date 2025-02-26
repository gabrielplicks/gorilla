import pickle



def serialize(obj: object) -> None:

    """Serializes a Python object using the pickle module.

    Args:

        obj: The Python object to serialize.

    """

    with open('serialized_obj.pkl', 'wb') as f:

        pickler = pickle.Pickler(f)

        pickler.dump(obj)



def unpickle() -> object:

    """Unpickles a Python object using the pickle module.

    Returns:

        The unpickled Python object.

    """

    with open('serialized_obj.pkl', 'rb') as f:

        unpickler = pickle.Unpickler(f)

        return unpickler.load()

