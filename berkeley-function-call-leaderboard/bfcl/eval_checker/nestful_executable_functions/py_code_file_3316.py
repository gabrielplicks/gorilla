import pickle



def sum_binary_file(file_path: str) -> int:

    """Calculates the sum of all integers in a binary file.



    Args:

        file_path: The path to the binary file.



    Returns:

        The sum of all integers in the binary file.

    """

    with open(file_path, "rb") as file:

        serialized_list = file.read()

        deserialized_list = pickle.loads(serialized_list)

        return sum(deserialized_list)

