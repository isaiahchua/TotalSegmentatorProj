import functools
import time

def PrintShapeDecorator(print_enabled):
    def PrintShape(func):
        def printing(*args, **kwargs):
            out = func(*args, **kwargs)
            if print_enabled:
                print(out.detach().shape)
            return out
        return printing
    return PrintShape

def TimeFuncDecorator(time_enabled):
    def TimeFunc(func):
        @functools.wraps(func)
        def Timer(*args, **kwargs):
            if time_enabled:
                start = time.time()
                func(*args, **kwargs)
                end = time.time()
                print("\n")
                start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
                end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
                time_taken = end - start
                if time_taken > 3600.:
                    divisor = 3600.
                    suffix = "hr"
                elif time_taken > 60:
                    divisor = 60.
                    suffix = "min"
                else:
                    divisor = 1.
                    suffix = "s"
                print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(time_taken)/divisor:.3f} {suffix}")
            else:
                func(*args, **kwargs)
        return Timer
    return TimeFunc

def PrintTimeDecorator(func):
    def PrintTime(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        print(f"time taken: {end - start} s")
        return results
    return TimeModule
