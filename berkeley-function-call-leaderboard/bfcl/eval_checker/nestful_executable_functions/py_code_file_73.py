from typing import List



def merge_lists_by_extensions(list1: List[str], list2: List[str]) -> List[str]:

    """Merges two lists of file names based on their file extensions.



    Args:

        list1: The first list of file names.

        list2: The second list of file names.



    Returns:

        A list of file names that have common extensions between the two input lists.

        The output list only includes the file names without the extensions.

    """

    extensions = set()

    result = []



    for filename in list1:

        extension = filename.split('.')[-1].lower()

        extensions.add(extension)



    for filename in list2:

        extension = filename.split('.')[-1].lower()

        if extension in extensions:

            result.append(filename.split('.')[0])



    return result

