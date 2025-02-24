from typing import List, Union



def get_float_elements(targetList: List[Union[int, float, str]]) -> List[float]:

    """

    Returns a list of float elements from the input list.

    If the input list contains non-float elements, they are converted to float.

    If the input list contains no float elements, an empty list is returned.



    Args:

        targetList: A list of any types of elements.

    """

    outputList = []

    for element in targetList:

        if not isinstance(element, float):

            element = float(element)

        outputList.append(element)

    return outputList

