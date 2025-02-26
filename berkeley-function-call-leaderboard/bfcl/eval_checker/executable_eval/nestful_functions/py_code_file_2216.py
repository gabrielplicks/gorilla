from collections import defaultdict

from typing import Dict, List



def group_fingerprints_by_pos_id(fingerprints: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:

    """Groups a list of WiFi fingerprint objects based on their position IDs.



    Args:

        fingerprints: A list of fingerprints, where each entry is a dictionary. Each fingerprint has a 'pos_id' key.



    Returns:

        A dictionary where each key is a position ID and the corresponding value is a list of fingerprints associated with that ID.

    """

    fingerprints_by_pos_id = defaultdict(list)

    for fingerprint in fingerprints:

        fingerprints_by_pos_id[fingerprint['pos_id']].append(fingerprint)

    return fingerprints_by_pos_id

