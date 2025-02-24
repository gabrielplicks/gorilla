from typing import List



def get_code_blocks_from_string(string: str) -> List[str]:

    """Extracts code blocks from a given string.



    Args:

        string: The input string to extract code blocks from.



    Returns:

        A list of code blocks found within the input string.

    """

    code_blocks = []



    def extract_code_blocks(s: str):

        start = s.find('{')

        end = s.find('}')



        if start != -1 and end != -1:

            code = s[start + 1:end]

            code_blocks.append(code)

            extract_code_blocks(s[end + 1:])



    extract_code_blocks(string)

    return code_blocks

