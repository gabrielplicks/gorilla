import itertools



def flatten_iterables(iterables: list) -> None:

    """Flattens a list of iterables into a generator object.

    Args:

        iterables: A list of iterables to be flattened.

    """

    yield from itertools.chain(*iterables)

