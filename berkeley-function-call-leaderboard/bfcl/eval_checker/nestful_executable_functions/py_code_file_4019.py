from typing import List



def process_text_file(file_path: str) -> str:

    """Reads a text file and processes its lines.



    Removes any empty lines, converts each line to lowercase, and joins the lines back into a single string.



    Args:

        file_path: The path to the text file.



    Returns:

        The processed text as a single string.

    """

    with open(file_path, "r") as file:

        lines: List[str] = file.readlines()



    processed_lines: List[str] = []

    for line in lines:

        line = line.strip()  # Remove leading and trailing whitespace

        if line:

            processed_lines.append(line.lower())



    return "".join(processed_lines)

