from typing import Tuple



def generate_ns_uuid(namespace: str, name: str) -> str:

    """Generates a UUID in the `namespace:` scheme.



    Args:

        namespace: The namespace of the UUID.

        name: The name of the UUID.



    Returns:

        A UUID in the `namespace:` scheme.

    """

    assert isinstance(namespace, str), "Namespace must be a string"

    assert isinstance(name, str), "Name must be a string"

    return f"{namespace}:{name}"

