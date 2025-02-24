import torch



def use_cuda() -> bool:

    """Checks if a CUDA device is available.



    Returns:

        True if a CUDA device is available, False otherwise.

    """

    return torch.cuda.is_available()



def use_cuda_if_available() -> bool:

    """Initializes the CUDA device and returns True if successful, False otherwise.



    Returns:

        True if the CUDA device is initialized successfully, False otherwise.

    """

    try:

        torch.cuda.init()

        return True

    except RuntimeError:

        return False

    except Exception:

        return None

