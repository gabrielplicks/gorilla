import struct



def read_serialized_list(file_path: str) -> list[int]:

    """Reads a binary file containing a list of integers, where each integer is serialized as four bytes.

    Args:

        file_path: The path to the binary file.

    Returns:

        A list of integers representing the serialized integers from the file.

    """

    with open(file_path, 'rb') as f:

        num_elements = struct.unpack('I', f.read(4))[0]

        serialized_list = []

        for _ in range(num_elements):

            serialized_list.append(struct.unpack('I', f.read(4))[0])



    return serialized_list

