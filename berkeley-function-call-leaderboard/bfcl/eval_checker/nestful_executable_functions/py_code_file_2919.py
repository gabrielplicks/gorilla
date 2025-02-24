from typing import Dict, List



def expand_scope(scope: Dict[str, str], props: List[str]) -> Dict[str, str]:

    """Expands a given scope dictionary with additional properties specified in a list.



    Args:

        scope: A dictionary representing the current scope.

        props: A list of property names to expand.



    Returns:

        A new scope dictionary containing all properties inherited from the current scope,

        plus additional properties that are explicitly specified in the props list.

        If a property is not found in the current scope, it is added to the new scope

        with a value of None.

    """

    new_scope = scope.copy()

    for prop in props:

        if prop not in new_scope:

            new_scope[prop] = None

    return new_scope

