import threading



def get_thread_id_str() -> str:

    """Returns the current thread ID as a string. If the current thread is the main thread, returns the string 'Main'.



    Args:

        None

    """

    thread = threading.current_thread()

    if thread.name == 'MainThread':

        return 'Main'

    else:

        return str(thread.ident)

