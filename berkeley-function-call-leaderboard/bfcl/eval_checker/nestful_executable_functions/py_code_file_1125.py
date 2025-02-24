from typing import AnyStr



def format_namespace(namespace: AnyStr) -> AnyStr:

    """Formats a fully-qualified namespace as a Python module name.

    Args:

        namespace: The fully-qualified namespace.

    """

    words = namespace.strip().split()

    words = [word.lower() for word in words]

    return "_".join(words)

