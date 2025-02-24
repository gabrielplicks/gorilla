from typing import List, Any, Union



def extract_attributes_from_objects(objects: List[Any], attributes: List[str], ignore_missing: bool = True) -> List[Union[Any, None]]:

    """Extracts the specified attributes from a list of objects and returns a list of extracted attribute values.



    Args:

        objects: A list of objects.

        attributes: A list of attribute names to extract.

        ignore_missing: A boolean flag indicating whether to ignore missing attributes (default to True).



    Returns:

        A list of extracted attribute values.

    """

    def extract_attribute(obj: Any, attr: str) -> Union[Any, None]:

        try:

            return getattr(obj, attr)

        except AttributeError:

            if ignore_missing:

                return None

            else:

                raise

    return [extract_attribute(obj, attr) for obj in objects for attr in attributes]

