from typing import List, Union



def concat_list_str(lst: Union[List[str], None], delimiter: str = None) -> str:

    """Concatenates a list of strings into a single string with a specified delimiter.



    Args:

        lst: The list of strings to concatenate.

        delimiter: The delimiter to use between elements. If not specified, a comma (,) is used.



    Returns:

        The concatenated string.

    """

    if not lst:

        return ''

    if delimiter is None:

        delimiter = ','

    return delimiter.join(lst)

