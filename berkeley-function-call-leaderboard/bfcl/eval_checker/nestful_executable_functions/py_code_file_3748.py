from typing import Dict, Any



def retrieve_word(node: Dict[str, Any], letters: str) -> str:

    """Retrieves the word associated with the node from a dictionary.



    Args:

        node: The node to retrieve the word from.

        letters: The string of letters to match with the node.



    Returns:

        The word associated with the node if it is a valid node, or None otherwise.

    """

    if not node or not node.get('value'):

        return None



    if node.get('data') and node.get('data').get('counter') > 0:

        if letters == '':

            return node['value']

        elif node.get('data') and node.get('data').get('children'):

            return retrieve_word(node.get('data').get('children').get(letters[0]), letters[1:])

    return None

