import hashlib



def compute_md5sum(file_path: str) -> str:

    """Computes the md5sum of a file at `file_path` without loading the entire file into memory.



    Args:

        file_path: The path to the file to compute the md5sum of.

    """

    md5 = hashlib.md5()



    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024)  # Read 1024 bytes at a time

            if not chunk:

                break

            md5.update(chunk)



    return md5.hexdigest()

