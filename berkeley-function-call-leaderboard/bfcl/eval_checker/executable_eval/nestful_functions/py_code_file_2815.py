import textwrap



def convert_to_fixed_length_records(text: str, record_length: int) -> list[str]:

    """Converts the given text to fixed-length records by padding with spaces or truncating the text to fit the record length.



    Args:

        text: The input text to be converted.

        record_length: The length of each record.

    """

    wrapped_lines = textwrap.TextWrapper(width=record_length).wrap(text)

    return [line.ljust(record_length) for line in wrapped_lines]

