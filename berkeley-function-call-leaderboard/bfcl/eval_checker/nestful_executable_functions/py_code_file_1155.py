from typing import List, Tuple



def transform_tuple_list(tuple_list: List[Tuple[str, str]]) -> str:

    """Transforms a list of tuples into a single string of the format "name1=value1, name2=value2, ..., nameN=valueN".

    The names should be in alphabetical order and separated by commas with no spaces.

    Args:

        tuple_list: A list of tuples, where each tuple represents a name-value pair.

    """

    transformed_list = []

    for name, value in tuple_list:

        transformed_list.append(f'{name}={value}')

    return ','.join(sorted(transformed_list))

