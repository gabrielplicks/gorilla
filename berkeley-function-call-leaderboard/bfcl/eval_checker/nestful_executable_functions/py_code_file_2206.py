from typing import Dict



def clean_mesh(mesh: Dict) -> Dict:

    """

    Validates a mesh and removes redundant elements from the mesh.



    Args:

        mesh: The input mesh to be cleaned.



    Raises:

        ValueError: If the mesh is not valid.

    """

    if not mesh.get("nodes"):

        raise ValueError("Invalid mesh: Missing 'nodes' property")

    seen = set()

    for element in mesh["elements"]:

        if element not in seen:

            seen.add(element)



    mesh["elements"] = list(seen)

    return mesh

