from typing import Union



def get_build_stage(build_number: Union[int, float]) -> str:

    """

    Returns the build stage corresponding to the given build number.



    Args:

        build_number: The build number.

    """

    if 0 <= build_number < 10:

        return "Alpha"

    elif 10 <= build_number < 50:

        return "Beta"

    elif 50 <= build_number < 100:

        return "Release Candidate"

    elif build_number == 100:

        return "Release"

    else:

        return "Unknown"

