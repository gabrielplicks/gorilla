from typing import List, Any



def get_attribute_values_for_entities(entities: List[Any], attribute_name: str) -> List[Any]:

    """

    Returns a list of the values of the specified attribute for all entities in the list that have that attribute.

    If the entity does not have the attribute, the corresponding value is None.



    Args:

        entities: A list of entities.

        attribute_name: The name of the attribute to retrieve.

    """

    attribute_values = []

    for entity in entities:

        value = getattr(entity, attribute_name, None)

        attribute_values.append(value)

    return attribute_values

