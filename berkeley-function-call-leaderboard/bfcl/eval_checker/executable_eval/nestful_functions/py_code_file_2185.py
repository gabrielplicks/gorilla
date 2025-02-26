import pickle



def serialize(obj: object, filename: str):

    """Serializes an object into a file using pickle.



    Args:

        obj: The object to serialize.

        filename: The name of the file to save the serialized object to.

    """

    with open(filename, 'wb') as file:

        pickle.dump(obj, file)



def deserialize(filename: str) -> object:

    """Deserializes an object from a file using pickle.



    Args:

        filename: The name of the file to load the serialized object from.



    Returns:

        The deserialized object.

    """

    with open(filename, 'rb') as file:

        obj = pickle.load(file)

    return obj

