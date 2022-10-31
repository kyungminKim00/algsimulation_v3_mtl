import pickle
import os


def check_model_pool(dir_name, target_name, forward_ndx):
    torken = "{}_T{}.pkl".format(target_name, forward_ndx)
    torken = "./save/model_repo_meta/{}".format(torken)

    if os.path.isfile(torken):
        with open(torken, "rb") as fp:
            repo_info = pickle.load(fp)
            fp.close()

        for it in repo_info:
            if it["m_name"] == dir_name.split("/")[-1]:
                # # keep final model only
                # for rm_file in os.listdir(dir_name):
                #     if '.pkl' in rm_file and not it['model_name'] in rm_file:
                #         os.remove(dir_name + '/' + rm_file)
                return False
    return True
