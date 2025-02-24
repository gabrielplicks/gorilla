import numpy as np



def serialize_numpy_array(arr: np.ndarray, path: str) -> bool:

    """

    Serializes a NumPy array to a custom format, where the first line contains a single integer specifying the number of rows in the array, and each subsequent line contains all the elements in a row separated by a space.

    Args:

        arr: The NumPy array to serialize.

        path: The path to the file where the serialized data should be written.

    """

    try:

        with open(path, 'w') as f:

            f.write(str(len(arr)) + '\n')

            for row in arr:

                f.write(' '.join(map(str, row)) + '\n')

    except Exception as e:

        print(f'Error serializing data: {e}')

        return False

    return True

