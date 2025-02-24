import re



def has_dependency_on_target(target: str, dep_target: str) -> bool:

    """Checks whether a given target exists in the dependencies of another target.



    Args:

        target: The target path.

        dep_target: The dependency path.



    Returns:

        True if the target exists in the dependencies of the other target, False otherwise.

    """

    with open("BUILD", "r") as build_file:

        build_contents = build_file.read()

    target_definitions = re.findall(r"(?<=target\()\".+\"", build_contents)

    for definition in target_definitions:

        if target in definition and dep_target in definition:

            return True



    return False

