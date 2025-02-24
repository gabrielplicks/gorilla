import pickle

import numpy as np



def load_data_from_file(file_name: str) -> np.ndarray:

    """Loads data from a file in pickle format and returns it as a numpy array.



    Args:

        file_name: The name of the file containing the data.

    """

    with open(file_name, 'rb') as file:

        data = pickle.load(file)



    return np.array(data)

