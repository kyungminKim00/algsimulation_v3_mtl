import pickle
from typing import Any, Literal

from gym.envs.registration import register

with open("./g", "rb") as fp:
    g: Any = pickle.load(fp)
    fp.close()
    for k, v in g.items():
        register(id=k, entry_point=v)


__version__: Literal["2.4.1"] = "2.4.1"
