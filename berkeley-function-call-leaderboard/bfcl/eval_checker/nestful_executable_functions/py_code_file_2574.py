from typing import List, Tuple



def select_resource(resources: List[Tuple[int, str]], target_name: str, idx: int = 0, output: List[str] = []) -> List[str]:

    """Finds the full resource name given a partial target name.



    Args:

        resources: A list of resource name/identifier pairs.

        target_name: The target name to find.

        idx: The index to track the current position in the `resources` list.

        output: A list to store the current partial names.



    Returns:

        A list of full resource names that match the target name.

    """

    if idx == len(resources):

        return output

    partial_name = resources[idx][1]

    if partial_name[:len(target_name)] == target_name:

        output.append(partial_name)

    return select_resource(resources, target_name, idx + 1, output)

