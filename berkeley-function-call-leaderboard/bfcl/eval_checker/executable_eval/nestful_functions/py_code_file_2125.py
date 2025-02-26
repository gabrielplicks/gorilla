from typing import List



def build_file_path(parameters: List[str]) -> str:

    """Builds a file path for a Cloud Asset.

    Args:

        parameters: A list of parameters containing the project name, the asset type, and the asset name.

    """

    return f"/projects/{parameters[0]}/assets/{parameters[1]}/{parameters[2]}"

