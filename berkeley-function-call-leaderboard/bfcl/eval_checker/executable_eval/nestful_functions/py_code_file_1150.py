from typing import Dict, Any



def filter_dict_by_class(my_dict: Dict[Any, Any], class_type: type) -> Dict[Any, Any]:

    """Filters a dictionary, excluding all entries whose values are not an instance of the specified class.



    Args:

        my_dict: The dictionary to filter.

        class_type: The class type to filter by.



    Returns:

        A new dictionary with the filtered entries.

    """

    return {k: v for k, v in my_dict.items() if isinstance(v, class_type)}

