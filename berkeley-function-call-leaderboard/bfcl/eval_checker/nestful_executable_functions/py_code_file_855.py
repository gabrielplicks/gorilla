import json

import zlib



def zip_json(json_document: str, key: str, new_value: str) -> bytes:

    """Zips a JSON document by changing the value of a key and compressing the resulting JSON string.

    Args:

        json_document: The JSON document to zip.

        key: The key to modify in the JSON document.

        new_value: The new value to assign to the key.

    """

    data = json.loads(json_document)

    data[key] = new_value

    json_string = json.dumps(data)

    compressed_data = zlib.compress(json_string.encode())

    return compressed_data

