import json

from typing import List, Dict



def extract_captions(json_file: str) -> List[str]:

    """Extracts the captions from a given JSON file.



    Args:

        json_file: The path to the JSON file containing the captions.



    Returns:

        A list of extracted captions, each of which is a string.

    """

    with open(json_file, 'r') as f:

        data = json.load(f)  # Load the JSON file as a Python dictionary



    captions = []

    for dictionary in data:

        for data_dict in dictionary['data']:

            captions.append(data_dict['caption'])



    return captions

