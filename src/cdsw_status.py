# import re
# from PIL import Image
# import glob
# import platform
# import header.index_forecasting.RUNHEADER as RUNHEADER
# import datetime
# import util
# from util import get_domain_on_CDSW_env

import random
import os
import sys
import numpy as np
import argparse
import pickle

domain_search_parameter = {
    "INX_20": 1,
    "KS_20": 1,
    "Gold_20": 1,
    "FTSE_20": 1,
    "GDAXI_20": 1,
    "SSEC_20": 1,
    "BVSP_20": 1,
    "N225_20": 1,
    "INX_60": 2,
    "KS_60": 2,
    "Gold_60": 2,
    "FTSE_60": 2,
    "GDAXI_60": 2,
    "SSEC_60": 2,
    "BVSP_60": 2,
    "N225_60": 2,
    "INX_120": 3,
    "KS_120": 3,
    "Gold_120": 3,
    "FTSE_120": 3,
    "GDAXI_120": 3,
    "SSEC_120": 3,
    "BVSP_120": 3,
    "N225_120": 3,
    "US10YT_20": 4,
    "GB10YT_20": 4,
    "DE10YT_20": 4,
    "KR10YT_20": 4,
    "CN10YT_20": 4,
    "JP10YT_20": 4,
    "BR10YT_20": 4,
    "US10YT_60": 5,
    "GB10YT_60": 5,
    "DE10YT_60": 5,
    "KR10YT_60": 5,
    "CN10YT_60": 5,
    "JP10YT_60": 5,
    "BR10YT_60": 5,
    "US10YT_120": 6,
    "GB10YT_120": 6,
    "DE10YT_120": 6,
    "KR10YT_120": 6,
    "CN10YT_120": 6,
    "JP10YT_120": 6,
    "BR10YT_120": 6,
}

ex_20 = [
    "INX_60",
    "KS_60",
    "Gold_60",
    "FTSE_60",
    "GDAXI_60",
    "SSEC_60",
    "BVSP_60",
    "N225_60",
    "INX_120",
    "KS_120",
    "Gold_120",
    "FTSE_120",
    "GDAXI_120",
    "SSEC_120",
    "BVSP_120",
    "N225_120",
    "US10YT_60",
    "GB10YT_60",
    "DE10YT_60",
    "KR10YT_60",
    "CN10YT_60",
    "JP10YT_60",
    "BR10YT_60",
    "US10YT_120",
    "GB10YT_120",
    "DE10YT_120",
    "KR10YT_120",
    "CN10YT_120",
    "JP10YT_120",
    "BR10YT_120",
]

ex_60 = [
    "INX_20",
    "KS_20",
    "Gold_20",
    "FTSE_20",
    "GDAXI_20",
    "SSEC_20",
    "BVSP_20",
    "N225_20",
    "INX_120",
    "KS_120",
    "Gold_120",
    "FTSE_120",
    "GDAXI_120",
    "SSEC_120",
    "BVSP_120",
    "N225_120",
    "US10YT_20",
    "GB10YT_20",
    "DE10YT_20",
    "KR10YT_20",
    "CN10YT_20",
    "JP10YT_20",
    "BR10YT_20",
    "US10YT_120",
    "GB10YT_120",
    "DE10YT_120",
    "KR10YT_120",
    "CN10YT_120",
    "JP10YT_120",
    "BR10YT_120",
]

ex_120 = [
    "INX_20",
    "KS_20",
    "Gold_20",
    "FTSE_20",
    "GDAXI_20",
    "SSEC_20",
    "BVSP_20",
    "N225_20",
    "INX_60",
    "KS_60",
    "Gold_60",
    "FTSE_60",
    "GDAXI_60",
    "SSEC_60",
    "BVSP_60",
    "N225_60",
    "US10YT_20",
    "GB10YT_20",
    "DE10YT_20",
    "KR10YT_20",
    "CN10YT_20",
    "JP10YT_20",
    "BR10YT_20",
    "US10YT_60",
    "GB10YT_60",
    "DE10YT_60",
    "KR10YT_60",
    "CN10YT_60",
    "JP10YT_60",
    "BR10YT_60",
]


def write_file(filename, val):
    fp = open(filename, "w")
    fp.write(val)
    fp.close()


def read_file(filename):
    fp = open(filename, "r")
    val = fp.readline().replace("\n", "")
    fp.close()
    return val


def validate(forward_ndx, is_update_20, is_update_60, is_update_120):
    if forward_ndx == 20:
        is_update_20 = True
    elif forward_ndx == 60:
        is_update_60 = True
    elif forward_ndx == 120:
        is_update_120 = True

    return is_update_20, is_update_60, is_update_120


def finalize_cdsw_status(is_update_20, is_update_60, is_update_120):
    if is_update_20 == False and is_update_60 == False and is_update_120 == False:
        _write_file(20, None)
        _write_file(60, None)
        _write_file(120, None)
        assert (
            False
        ), "Increase the value one of variables UP_TO_20, UP_TO_60, UP_TO_120"
    else:
        if not is_update_20:
            _write_file(20, pop_market(ex_20))
        if not is_update_20:
            _write_file(60, pop_market(ex_60))
        if not is_update_20:
            _write_file(120, pop_market(ex_120))
    current_learning_model()


def current_learning_model():
    print("Current learning model:")
    print("{} from cdsw_20.txt".format(read_file("./cdsw_20.txt")))
    print("{} from cdsw_60.txt".format(read_file("./cdsw_60.txt")))
    print("{} from cdsw_120.txt".format(read_file("./cdsw_120.txt")))


def read_pickle(file_name):
    fp = open(file_name, "rb")
    data = pickle.load(fp)
    fp.close()
    return data


def pop_market(market_pool):
    while True:
        market = market_pool[np.random.randint(len(market_pool))]
        forward_ndx = int(market.split("_")[1])
        file_name = "_T".join(market.split("_"))

        data = read_pickle("./save/model_repo_meta/{}.pkl".format(file_name))
        if create_new_job(data, forward_ndx):
            return market


def _write_file(forward_ndx, it):
    write_file("./cdsw_{}.txt".format(str(forward_ndx)), it)
    write_file("./cdsw_status.txt", "system_busy")


def create_new_job(data, forward_ndx):
    if forward_ndx == 20:
        return len(data) == (UP_TO_20 - 1)
    elif forward_ndx == 60:
        return len(data) == (UP_TO_60 - 1)
    elif forward_ndx == 120:
        return len(data) == (UP_TO_120 - 1)


UP_TO_20, UP_TO_60, UP_TO_120 = 2, 2, 2
if __name__ == "__main__":
    is_update_20, is_update_60, is_update_120 = False, False, False

    parser = argparse.ArgumentParser("")
    parser.add_argument("--update_system_status", type=int, default=0)
    args = parser.parse_args()

    if args.update_system_status:  # auto clean 이후 시작
        write_file("./cdsw_status.txt", "system_idle")
        print("system_idle")
    else:  # script_all_in_one 앞에 시작
        system_idel = read_file("./cdsw_status.txt")
        if system_idel == "system_idle":
            sl = list(domain_search_parameter.keys())
            random.shuffle(sl)
            for it in sl:
                _, forward_ndx = it.split("_")
                cond = "_T".join(it.split("_"))
                forward_ndx = int(forward_ndx)
                _file_name = "./save/model_repo_meta/{}.pkl".format(cond)
                if os.path.isfile(_file_name):
                    data = read_pickle(_file_name)
                    if create_new_job(data, forward_ndx):
                        _write_file(forward_ndx, it)
                        is_update_20, is_update_60, is_update_120 = validate(
                            forward_ndx, is_update_20, is_update_60, is_update_120
                        )
                    else:
                        pass
                else:
                    _write_file(forward_ndx, it)
                    is_update_20, is_update_60, is_update_120 = validate(
                        forward_ndx, is_update_20, is_update_60, is_update_120
                    )
            finalize_cdsw_status(is_update_20, is_update_60, is_update_120)
        else:
            current_learning_model()
            print("system_busy")
            sys.exit()
