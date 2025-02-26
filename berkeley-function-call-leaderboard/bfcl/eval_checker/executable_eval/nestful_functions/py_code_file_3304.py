from typing import Dict



def filter_metadata(meta: Dict[str, str]) -> Dict[str, str]:

    """

    Given a dictionary of metadata `meta` about a file, return a new dictionary containing only the metadata fields starting with `dc_`.



    Args:

        meta: The dictionary of metadata about a file.

    """

    return {key: meta[key] for key in meta if key.startswith("dc_")}

