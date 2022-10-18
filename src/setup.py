from distutils.core import setup
from Cython.Build import cythonize
import os
import platform

t_dirs = [
    "./",
    "./custom_model/index_forecasting/a2c",
    "./custom_model/index_forecasting/nets",
    "./custom_model/index_forecasting/policies",
    "./custom_model/market_timing/a2c",
    "./custom_model/market_timing/nets",
    "./custom_model/market_timing/policies",
    "./envs",
    "./datasets",
]

file_lists = [
    "./index_forecasting_select_model.pyx",
    "./index_forecasting_adhoc.pyx",
    "./market_timing_select_model.pyx",
    "./market_timing_adhoc.pyx",
    "./auto_clean_envs.pyx",
    # './custom_model/index_forecasting/a2c/*.pyx',
    # './custom_model/index_forecasting/nets/*.pyx',
    # './custom_model/index_forecasting/policies/*.pyx',
    "./envs/*.pyx",
    "./datasets/*.pyx",
]

finalize = False
if finalize:
    for dir in t_dirs:
        for fn in os.listdir(dir):
            tmp_src = os.path.join(dir, fn)
            if platform.system() == "Windows":
                tmp_src = tmp_src.replace("\\", "/")

            if ".pyx" == fn[-4:] or ".c" == fn[-2:]:
                os.remove(tmp_src)

if __name__ == "__main__":
    setup(
        ext_modules=cythonize(file_lists, compiler_directives={"language_level": 2})
        # ext_modules = cythonize(file_lists, compiler_directives={'language_level': 2}, gdb_debug=True)
    )
