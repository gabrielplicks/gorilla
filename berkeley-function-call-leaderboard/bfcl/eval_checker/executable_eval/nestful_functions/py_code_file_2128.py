from typing import Dict



def concatenate_kwargs(

    **kwargs: str,

) -> str:

    """Concatenates the values of keyword arguments into a single string, using their corresponding keys as the label.

    Args:

        kwargs: The keyword arguments to be concatenated.

    """

    return ", ".join(f"{key}: {value}" for key, value in kwargs.items())

