from typing import List



def make_python_code(texts: List[str]) -> str:

    """Creates a Python script with print statements for each string in the input list.



    Each print statement displays the index of the string in the input list and the string itself, using string interpolation.



    Args:

        texts: A list of strings.



    Returns:

        A Python script as a string.

    """

    script = []

    for i, text in enumerate(texts):

        script.append(f'print("Index {i}: {text}")\n')

    return "".join(script)

