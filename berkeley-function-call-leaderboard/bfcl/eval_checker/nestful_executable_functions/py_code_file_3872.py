import pickle

import os

from typing import List, Dict



def process_pickle(pickle_path: str) -> List[Dict[str, str]]:

    """

    Processes a pickle file containing a list of dictionaries and returns a list of dictionaries with two additional keys: `timestamp` and `source`.



    Args:

        pickle_path: The file path to the pickle file.



    Returns:

        A list of dictionaries with two additional keys: `timestamp` and `source`.

    """

    with open(pickle_path, "rb") as f:

        data = pickle.load(f)



    for d in data:

        d["timestamp"] = d.get("timestamp", None)

        d["source"] = os.path.splitext(os.path.basename(pickle_path))[0]



    return data

