import gzip

import os

from typing import BinaryIO



def decompress_gzip(gz_filepath: str, out_dir: str) -> None:

    """Decompresses a gzipped file into the provided directory.

    Args:

        gz_filepath: The path to the gzipped file.

        out_dir: The path to the output directory.

    Raises:

        FileExistsError: If the output directory already exists.

        Exception: If an error occurs during decompression.

    """

    if not os.path.exists(out_dir):

        os.makedirs(out_dir)

    else:

        raise FileExistsError(f"The directory {out_dir} already exists")



    try:

        with gzip.open(gz_filepath, 'rb') as f_in, open(out_dir, 'wb') as f_out:

            f_out.write(f_in.read())

    except Exception as e:

        print(f"Error: {e}")

        raise

