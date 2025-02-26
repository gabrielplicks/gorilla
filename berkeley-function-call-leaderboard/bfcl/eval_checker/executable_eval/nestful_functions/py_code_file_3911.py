import ast



def serialize_tuple(tup: tuple) -> str:

    """Serializes a Python tuple as a string representation of a Python tuple literal.



    Args:

        tup: The tuple to serialize.



    Returns:

        The string representation of the tuple.

    """

    assert isinstance(tup, tuple), "Input must be a tuple"

    return ast.literal_eval(repr(tup))

