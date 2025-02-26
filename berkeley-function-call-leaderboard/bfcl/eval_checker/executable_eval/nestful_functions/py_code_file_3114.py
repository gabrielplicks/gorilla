import json



def json_logger(input_json: dict) -> str:

    """Serializes a JSON object and logs the output to a file.



    Args:

        input_json: The input JSON object.



    Returns:

        The serialized JSON string.

    """

    serialized_json = json.dumps(input_json)

    with open("log_file.txt", "w") as log_file:

        log_file.write(serialized_json)

    return serialized_json

