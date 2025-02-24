from typing import Dict, List, Union



def find_variables_in_scope(scoperef: Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]], scope: str) -> Union[List[Dict[str, str]], None]:

    """Find variables in a given scope.



    Args:

        scoperef: The scoperef object containing scope information and variable mappings.

        scope: The name of the scope to find variables in.



    Returns:

        A list of variable mappings associated with the given scope, or None if no variables are found.

    """

    if scoperef['name'] == scope:

        return scoperef['var_mappings']

    for ref in scoperef['scoperefs']:

        variables = find_variables_in_scope(ref, scope)

        if variables:

            return variables

    return None

