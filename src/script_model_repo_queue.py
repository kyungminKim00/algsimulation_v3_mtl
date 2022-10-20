import pickle
import os
import sys
from util import read_pickle, write_pickle, str_join, remove_duplicaated_dict_in_list

if __name__ == "__main__":
    source_dir_1 = "./save/model_repo_meta"
    source_dir_2 = "./save/model_repo_meta_queue"
    target_dir = "./save/model_repo_meta_queue/final"

    for it_1 in os.listdir(source_dir_1):
        for it_2 in os.listdir(source_dir_2):
            if it_1 == it_2:
                md1 = read_pickle(str_join('/', source_dir_1, it_1))
                md2 = read_pickle(str_join('/', source_dir_2, it_2))
                target_model = md2 + md1
                target_model = remove_duplicaated_dict_in_list(target_model)
                write_pickle(target_model, str_join('/', target_dir, it_1))
                print('[{}] has been updatated with [{}] on [{}]'.format(
                    str_join('/', source_dir_2, it_2), 
                    str_join('/', source_dir_1, it_1), 
                    str_join('/', target_dir, it_1)))
                
