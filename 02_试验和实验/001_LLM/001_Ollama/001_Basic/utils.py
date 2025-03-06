from time import perf_counter


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} 耗时：{end - start:.6f} 秒")
        return result
    return wrapper