import os
import argparse
import util
import header.index_forecasting.RUNHEADER as RUNHEADER
import shutil
import platform
from util import funTime, discount_reward, writeFile, loadFile


def recent_procedure(file_name, process_id, mode):
    json_file_location = ''
    with open('{}{}.txt'.format(file_name, str(process_id)), mode) as _f_out:
        if mode == 'w':
            print(RUNHEADER.m_name, file=_f_out)
        elif mode == 'r':
            json_file_location = _f_out.readline()
        else:
            assert False, '<recent_procedure> : mode error'
        _f_out.close()
    return json_file_location.replace('\n', '')


# if platform.system() == 'Windows':
#     current_root = os.getcwd() + '\\save'
# else:  # assume ubuntu
#     current_root = os.getcwd() + '/save'

parser = argparse.ArgumentParser('')
# parser.add_argument('--process_id', type=int, required=True)
parser.add_argument('--process_id', type=int, default=0)
args = parser.parse_args()

json_location = recent_procedure('./agent_log/buffer_generate_model_p', args.process_id, 'r')
json_location = './save/model/rllearn/' + json_location
dict_RUNHEADER = util.json2dict('{}/agent_parameter.json'.format(json_location))
# re-load
for key in dict_RUNHEADER.keys():
    RUNHEADER.__dict__[key] = dict_RUNHEADER[key]
target_folder = RUNHEADER.m_offline_buffer_file

# merge buffer
file_names = os.listdir(target_folder)
buffer_names = [buffer_name for buffer_name in file_names if '.pkl' and 'buffer_' in buffer_name]
buffer = None
total_samples = None

assert len(buffer_names) > 1, 'There is no file to merge'
for buffer_name in buffer_names:
    buffer_file_name = target_folder + '/' + buffer_name[:-4]
    if buffer is None:
        buffer = loadFile(buffer_file_name)
    else:
        buffer.append(loadFile(buffer_file_name))
total_samples = (buffer.num_in_buffer // 7) + 1

writeFile('{}/cloud_buffer_E{}_S{}_U{}'.format(target_folder, RUNHEADER.m_n_cpu, RUNHEADER.m_n_step, total_samples), buffer)
