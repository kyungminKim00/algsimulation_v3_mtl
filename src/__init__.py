from gym.envs.registration import register
import pickle

with open("./g", "rb") as fp:
    g = pickle.load(fp)
    fp.close()
[
    register(
        id=k,
        entry_point=v,
    )
    for k, v in g.items()
]

__version__ = "2.4.1"
