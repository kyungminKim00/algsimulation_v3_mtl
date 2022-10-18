from memory_profiler import profile
import numpy as np


@profile
def my_func():
    tmp = np.ones(1000000)


# my_func()
# my_func()
# my_func()
