from typing import List



def count_states_in_state_list(state_list: List[str]) -> int:

    """Calculates the number of unique states present in a list of states.



    Args:

        state_list: A list of strings that represent the state names.



    Returns:

        The number of unique states present in the list.

    """

    unique_states = set()



    for state in state_list:

        unique_states.add(state)



    return len(unique_states)

