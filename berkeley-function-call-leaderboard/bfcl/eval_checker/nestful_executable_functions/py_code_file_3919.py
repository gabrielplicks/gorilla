import json



def dump_to_json(dictionary: dict, file_path: str) -> bool:

    """Dumps a dictionary to a JSON file.



    Args:

        dictionary: The dictionary to be dumped.

        file_path: The path to the file where the dictionary will be dumped.



    Returns:

        A boolean indicating if the operation was successful or not.

    """

    try:

        with open(file_path, 'w') as f:

            json.dump(dictionary, f)

        return True

    except Exception as e:

        print(f"Error dumping dictionary to JSON: {e}")

        return False

