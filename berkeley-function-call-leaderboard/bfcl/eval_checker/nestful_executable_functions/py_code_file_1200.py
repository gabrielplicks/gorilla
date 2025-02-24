import json



def encode_data_objects(data_objects: list[dict]) -> bytes:

    """Encodes a list of dictionaries representing data objects into a bytes object.



    Args:

        data_objects: A list of dictionaries representing data objects.



    Returns:

        A bytes object containing the encoded data.

    """

    encoded_data = json.dumps(data_objects).encode('utf-8')

    return encoded_data

