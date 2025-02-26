import json

import numpy as np



def load_data_json(file_path: str) -> np.ndarray:

    """Loads the data field from a JSON file as a numpy array.



    Args:

        file_path: The path to the JSON file.



    Returns:

        The data field from the JSON file as a numpy array.

    """

    with open(file_path, 'r') as f:

        data = json.load(f)

    data = data['data']

    arr = np.array(data)



    return arr

