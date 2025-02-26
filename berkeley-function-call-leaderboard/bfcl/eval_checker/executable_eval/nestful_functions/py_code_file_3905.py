from typing import List, Dict



class Build:

    def __init__(self, id: int, name: str, status: str, start_time: str):

        self.id = id

        self.name = name

        self.status = status

        self.start_time = start_time



def group_builds_by_status(builds: List[Build]) -> Dict[str, List[Build]]:

    """Groups a list of Build objects by their status.



    Args:

        builds: A list of Build objects.



    Returns:

        A dictionary where the keys are the build statuses and the values are lists of Build objects with that status.

    """

    grouped_builds = {}



    for build in builds:

        if build.status not in grouped_builds:

            grouped_builds[build.status] = []

        grouped_builds[build.status].append(build)



    return grouped_builds

