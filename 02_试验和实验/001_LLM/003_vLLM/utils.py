from time import perf_counter


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} 耗时：{end - start:.6f} 秒")
        return result
    return wrapper


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"==== Prompt ====\n\n{prompt}\n\n==== Generated text ====\n\n{generated_text}")
    print("-" * 80)
