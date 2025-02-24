from typing import List, Dict



def find_packages_without_dependents(packages: List[Dict[str, List[str]]]) -> List[str]:

    """Finds packages that have no dependents.



    Args:

        packages: A list of packages where each package is represented as a dictionary with the keys `name`, `dependencies`, and `dependents`.



    Returns:

        A list of package names that do not have any dependents.

    """

    dependents_mapping = {}

    for package in packages:

        for dependent in package["dependents"]:

            if dependent not in dependents_mapping:

                dependents_mapping[dependent] = []

            dependents_mapping[dependent].append(package["name"])

    result = []

    for package in packages:

        if package["name"] not in dependents_mapping:

            result.append(package["name"])



    return result

