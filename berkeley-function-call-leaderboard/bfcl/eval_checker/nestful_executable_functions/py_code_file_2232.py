from typing import Dict



def check_pipeline_io_types(pipeline: Dict[str, str], pipeline_io_types: Dict[str, str]) -> bool:

    """Checks if the pipeline input and output types match the expected types defined in pipeline.



    Args:

        pipeline: A dictionary that defines the input and output types of a pipeline.

        pipeline_io_types: A dictionary that defines the expected input and output types of a pipeline.



    Returns:

        True if the pipeline input and output types match the expected types, False otherwise.

    """

    for key, value in pipeline.items():

        if key in pipeline_io_types and pipeline_io_types[key] != value:

            return False



    return True

