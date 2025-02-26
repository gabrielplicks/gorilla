import os



def process_file_path(input_file_path: str) -> str:

    """

    Processes a file path by replacing the file extension with "_processed" + file_extension.



    Args:

        input_file_path: The input file path.



    Returns:

        The processed file path.

    """

    directory_path, file_name = os.path.split(input_file_path)

    file_base_name, file_extension = os.path.splitext(file_name)

    processed_file_name = f"{file_base_name}_processed{file_extension}"

    processed_file_path = os.path.join(directory_path, processed_file_name)



    return processed_file_path

