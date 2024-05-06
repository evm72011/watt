import numpy as np
from time import time
from warnings import deprecated

@deprecated
def validate_size(size: int):
    def decorator(func):
        def wrapper(w: np.ndarray):
            if not w.size == size:
                raise Exception(f'Arg must have size={size}')
            return func(w)
        return wrapper
    return decorator


def check_arr_size(func):
    def wrapper(*args: np.ndarray):
        sizes = set(map(lambda x: x.size, args))
        assert len(sizes) == 1, 'Arguments must have the same size'
        return func(*args)
    return wrapper


def timeit(func):
    def wrapper(*args, **kw):
        start = time()
        result = func(*args, **kw)
        duration = 1000 * (time() - start)
        print(f'it took {duration:.3f} ms.')
        return result
    return wrapper


if __name__ == '__main__':
    @check_arr_size
    def foo(x: np.ndarray, y: np.ndarray ) -> None:
        pass

    foo(np.ones(2), np.ones(3))
