import asyncio



async def echo_event(input_string: str, delay: int) -> str:

    """

    An event-driven echo service that accepts an input string and returns the string after an artificial delay.

    Args:

        input_string: The input string to be echoed.

        delay: The artificial delay in seconds.

    Returns:

        The echoed string.

    """

    try:

        await asyncio.sleep(delay)

        return f"Echoed: {input_string}"

    except Exception as e:

        print(f"Exception occurred: {e}")

        return "Default value"

