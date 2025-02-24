from typing import List, Dict



def get_entity_ids(entities: List[Dict[str, str]]) -> List[str]:

    """Retrieves the IDs of a list of entities.



    Args:

        entities: A list of dictionaries representing entities. Each entity should have a 'kind' key that can be either 'user' or 'organization'. If the 'kind' is 'user', then the entity should have an 'id' key. Otherwise, if the 'kind' is 'organization', the entity should have an 'org_id' key instead.



    Returns:

        A list of entity IDs.

    """

    return [entity['id'] if entity['kind'] == 'user' else entity['org_id'] for entity in entities]

