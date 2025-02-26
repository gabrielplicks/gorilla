import re

from typing import AnyStr



def abbreviate(text: AnyStr) -> AnyStr:

    """Abbreviates a string by reducing any word with more than 10 characters to its first 10 characters.



    Args:

        text: The input string.



    Returns:

        The abbreviated string.

    """

    pattern = re.compile(r"\b\w{11,}\b")

    abbreviated_text = pattern.sub(lambda m: m.group()[:10], text)



    return abbreviated_text



from pathlib import Path

data_dir = Path("data")



def test_abbreviate():

    for filename in data_dir.glob("*.txt"):

        with open(filename, "r") as file:

            content = file.read()

        abbreviated_content = abbreviate(content)

        assert all(len(word) <= 10 for word in abbreviated_content.split())

