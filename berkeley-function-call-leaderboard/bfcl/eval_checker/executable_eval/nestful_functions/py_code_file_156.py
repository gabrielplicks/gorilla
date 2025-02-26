from typing import Dict, List



def get_version_files(version_dict: Dict[str, Dict[str, str]], version: str) -> List[str]:

    """

    Returns a list of filenames associated with a specific version.



    Args:

        version_dict: A dictionary where the keys are version names and the values are dictionaries containing the version's name, files, and description.

        version: The version to get the files for.



    Returns:

        A list of filenames associated with the specified version. If the version does not exist in the version_dict, an empty list is returned.

    """

    if version in version_dict:

        return version_dict[version]["files"]

    else:

        return []

