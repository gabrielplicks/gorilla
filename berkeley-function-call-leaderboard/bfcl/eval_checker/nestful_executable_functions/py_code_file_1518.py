from typing import List



def generate_names(prefix: str, count: int) -> List[str]:

    """Generates a list of `count` unique names, starting with `prefix` and followed by a numeric suffix.



    Args:

        prefix: The prefix to use for the names.

        count: The number of unique names to generate.



    Returns:

        A list of `count` unique names.

    """

    names = []

    counter = 0



    while len(names) < count:

        name = f"{prefix}{counter}"

        if name not in names:

            names.append(name)

            counter += 1

        else:

            counter += 1



    return names

