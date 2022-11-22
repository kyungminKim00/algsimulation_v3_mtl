import _pickle as pickle
import joblib
import numpy as np

from util import funTime


@funTime("pickle_test")
def pickle_test(data):
    with open("./test.pkl", "wb") as fp:
        pickle.dump(data, fp, protocol=5)
        fp.close()


@funTime("joblib_test")
def joblib_test(data):
    with open("./test_joblib.pkl", "wb") as fp:
        joblib.dump(data, fp)
        fp.close()


data = np.zeros([70000, 90000])
joblib_test(data)
pickle_test(data)
