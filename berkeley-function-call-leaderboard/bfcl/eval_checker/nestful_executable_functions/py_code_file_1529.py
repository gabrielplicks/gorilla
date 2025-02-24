import struct



def get_image_size(png_file_path: str) -> tuple:

    """Extracts the size of an image specified in a PNG file's header.



    Args:

        png_file_path: The path to the PNG file.



    Returns:

        A tuple containing the width and height of the image.

    """

    with open(png_file_path, 'rb') as f:

        header = f.read(8)

        if header[:8] != b'\x89PNG\r\n\x1a\n':

            raise Exception('Invalid PNG file')

        width_bytes = f.read(4)

        height_bytes = f.read(4)

        width = struct.unpack('>I', width_bytes)[0]

        height = struct.unpack('>I', height_bytes)[0]



        return width, height

