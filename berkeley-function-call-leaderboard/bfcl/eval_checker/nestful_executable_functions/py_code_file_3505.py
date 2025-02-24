from typing import Union



def check_logical_implication(premise: Union[bool, int], conclusion: Union[bool, int]) -> bool:

    """Checks logical implications using the rules mentioned above.

    Args:

        premise: The premise of the logical implication.

        conclusion: The conclusion of the logical implication.

    """

    if premise == False:

        return True

    elif premise == True and conclusion == True:

        return True

    else:

        return False

