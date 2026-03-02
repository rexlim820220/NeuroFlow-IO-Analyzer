import time
import functools

def connection_guard(func):
    """decorator: make sure exception handle during connection, and document the elapse period"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"[Log] {func.__name__} success, execute time: {duration:.4f}s")
            return result
        except:
            print(f"[Error] {func.__name__} exception")
            return None
    return wrapper

class PLCHandler:
    @connection_guard
    def read_data(self):
        pass

