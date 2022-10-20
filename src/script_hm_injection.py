import re
import os
from PIL import Image
import glob
import platform
import numpy as np
import header.index_forecasting.RUNHEADER as RUNHEADER
import argparse
import datetime
import util
import pickle

dataset_version_dict = {
    0: "v11",
    1: "v12",
    2: "v13",
    3: "v14",
    4: "v15",
    5: "v16",
    6: "v17",
    7: "v18",
    8: "v19",
    9: "v20",
    10: "v21",
    11: "v22",
    12: "v23",
    13: "v24",
    14: "v25",
}
mkidx_mkname = {
    0: 'INX',
    1: 'KS',
    2: 'Gold',
    3: 'US10YT',
    4: 'FTSE',
    5: 'GDAXI',
    6: 'SSEC',
    7: 'BVSP',
    8: 'N225',
    9: 'GB10YT',
    10: 'DE10YT',
    11: 'KR10YT',
    12: 'CN10YT',
    13: 'JP10YT',
    14: 'BR10YT',
}
mkname_mkidx = {v: k for k, v in mkidx_mkname.items()}

def load(filepath, method):
    with open(filepath, "rb") as fs:
        if method == "pickle":
            data = pickle.load(fs)
    fs.close()
    return data


def save(filepath, method, obj):
    with open(filepath, "wb") as fs:
        if method == "pickle":
            pickle.dump(obj, fs)
    fs.close()


def get_model_name(model_dir, model_name):
    search = "./save/model/rllearn/{}".format(model_dir)
    for it in os.listdir(search):
        if model_name in it:
            return it


def get_header_info(json_location):
    return util.json2dict("{}/agent_parameter.json".format(json_location))


def script_hm_injection(
    m_target_index, forward_ndx, dataset_version, base_dir, model_name
):
    target_name = RUNHEADER.target_id2name(m_target_index)
    domain_detail = "{}_T{}_{}".format(target_name, forward_ndx, dataset_version)
    domain = "{}_T{}".format(target_name, forward_ndx)
    # src_dir = './save/result/selected/{}/final'.format(domain_detail)
    src_dir = None
    target_file = "./save/model_repo_meta/{}.pkl".format(domain)
    time_now = (
        str(datetime.datetime.now())[:-10]
        .replace(":", "-")
        .replace("-", "")
        .replace(" ", "_")
    )

    header = get_header_info("./save/model/rllearn/{}".format(base_dir))
    init_repo_model = 1
    meta = {
        "domain_detail": domain_detail,
        "domain": domain,
        "src_dir": src_dir,
        "target_file": target_file,
        "base_dir": base_dir,
        "model_name": get_model_name(
            base_dir,
            "sub_epo_{}".format(model_name.split("sub_epo_")[1:][0].split("_")[0]),
        ),
        "create_date": time_now,
        "target_name": header["target_name"],
        "m_name": header["m_name"],
        "dataset_version": header["dataset_version"],
        "m_offline_buffer_file": init_repo_model,
        "latest": True,
        "current_period": False,  # historical model
    }
    if os.path.isfile(target_file):
        meta_list = load(target_file, "pickle")
        # mark latest model
        for idx in range(len(meta_list)):
            meta_list[idx]["latest"] = False
            meta_list[idx]["current_period"] = False
        meta_list.append(meta)
        save(target_file, "pickle", meta_list)
    else:
        meta_list = list()
        meta_list.append(meta)
        save(target_file, "pickle", meta_list)


if __name__ == "__main__":
    """configuration"""
    parser = argparse.ArgumentParser("")
    # # init args
    # parser.add_argument("--base_dir", type=str, required=True)
    # parser.add_argument("--model_name", type=str, required=True)
    # Demo    
    parser.add_argument(
        "--base_dir",
        type=str,
        default="IF_INX_T120_20210610_0423_m5_3_v11_20210610_0424_1131",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="20210610_0510_sub_epo_704_pe1.2_pl1.07_vl0.343_ev0.993.pkl",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    model_name = args.model_name
    m_target_index = mkname_mkidx[base_dir.split('_')[1]]
    forward_ndx = int(base_dir.split('_')[2][1:])
    
    script_hm_injection(
        m_target_index,
        forward_ndx,
        dataset_version_dict[m_target_index],
        base_dir,
        model_name,
    )
    print(
        "[{}] {} has been injected (m_target_index: {} forward_ndx: {})".format(
            base_dir,
            model_name,
            m_target_index,
            forward_ndx,
        )
    )
