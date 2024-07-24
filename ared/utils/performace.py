import time

def timeit(func):
    """

    Parameters
    ----------
    func :
        A function that needs to be timed

    Returns
        A wrapped function that princts the time it took for the execution of that function
    """
    def wrapper(*args, **kwargs):
        """

        Parameters
        ----------
        *args : parameters to the function that needs to be wrapped
            
        **kwargs : keyword arguments of the function that needs to be wrapped
            

        Returns
         the time decorator for the original function
        
        """
        start = time.time()
        result = func(*args, **kwargs)
        print(f'time taken by {func.__name__} is {time.time() - start}')
        return result
    return wrapper