from collections import OrderedDict

from typing import Dict



def get_project_ids(projects: Dict[str, Dict[str, any]]) -> List[int]:

    """Extracts the project IDs from a dictionary of projects.



    Args:

        projects: A dictionary of projects. Each project is a dictionary with keys "id", "name", "status", and "tasks".



    Returns:

        A list of project IDs.

    """

    project_ids = [project["id"] for project in projects.values()]

    return [id for id in OrderedDict.fromkeys(project_ids)]

