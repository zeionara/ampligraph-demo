import functools
import time


def measure_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Completed in {stringify_elapsed_time(time.time() - start)}")
        return result

    return wrapper


def stringify_elapsed_time(elapsed_time):
    return (
        f'{elapsed_time:.3f} seconds' if elapsed_time < 60 else
        f'{elapsed_time / 60:.3f} minutes' if elapsed_time < 3600 else
        f'{elapsed_time / 3600:.3f} hours'
    )
