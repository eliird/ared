import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'time taken by {func.__name__} is {time.time() - start}')
        return result
    return wrapper