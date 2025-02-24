from typing import Dict



def simplify_conditional(condition: Dict[str, bool], lookup_table: Dict[str, str]) -> str:

    """Simplifies an if-else conditional statement.



    Args:

        condition: A dictionary of boolean conditions.

        lookup_table: A dictionary of action keys.



    Returns:

        A string representing the simplified if-else conditional statement.

    """

    simplified = []

    for key, value in condition.items():

        if value and lookup_table.get(key) not in simplified:

            simplified.append(lookup_table[key])

        elif not value and lookup_table[key] in simplified:

            simplified.remove(lookup_table[key])



    if len(simplified) == len(lookup_table):

        return 'do_something'

    elif len(simplified) == 0:

        return 'do_nothing'

    else:

        return ' or '.join(simplified)

