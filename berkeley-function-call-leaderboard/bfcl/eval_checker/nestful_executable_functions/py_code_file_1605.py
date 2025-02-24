import json

from typing import List



def serialize_deserialize_int_list(int_list: List[int]) -> List[int]:

    """Serializes and deserializes a list of integers to and from a JSON file.

    Args:

        int_list: The list of integers to serialize and deserialize.

    Returns:

        The deserialized list of integers.

    """

    try:

        with open("int_list.json", "w") as file:

            json.dump(int_list, file, indent=2)

        with open("int_list.json", "r") as file:

            deserialized_list = json.load(file)



        return deserialized_list

    except Exception as e:

        print(f"Exception occurred: {e}")

        return []

