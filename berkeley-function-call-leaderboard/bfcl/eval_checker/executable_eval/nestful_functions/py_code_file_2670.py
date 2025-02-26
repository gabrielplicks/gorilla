import torch



def check_and_load_to_gpu(model: torch.nn.Module) -> None:

    """Checks if a PyTorch model has been loaded to a GPU device. If not, loads the model to the GPU.



    Args:

        model: The PyTorch model to be checked and loaded to the GPU.

    """

    if torch.cuda.is_available():

        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:

            torch.cuda.set_device(0)

        model.cuda()

