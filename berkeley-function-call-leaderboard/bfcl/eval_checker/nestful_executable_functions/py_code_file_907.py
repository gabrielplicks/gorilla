from typing import Union



def replace_col_prefix(col_name: str) -> Union[str, None]:

    """Replaces the 'col_' prefix of a column name with 'new_col_'.



    Args:

        col_name: The column name to be modified.



    Returns:

        The modified column name if the original column name starts with 'col_',

        otherwise the original column name.

    """

    if col_name.startswith('col_'):

        return f'new_{col_name}'

    else:

        return col_name

