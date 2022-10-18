import os
import argparse

import header.market_timing.RUNHEADER as RUNHEADER
if RUNHEADER.release:
    from libs import auto_clean_envs
else:
    import auto_clean_envs
import util
from util import get_domain_on_CDSW_env
import shutil
import platform
import sc_parameters as scp

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


if __name__ == '__main__':
    try:
        if platform.system() == 'Windows':
            current_root = os.getcwd() + '\\save'
        else:  # assume ubuntu
            current_root = os.getcwd() + '/save'

        parser = argparse.ArgumentParser('')
        # init args
        parser.add_argument('--process_id', type=int, default=None)
        parser.add_argument('--m_target_index', type=int, default=None)
        parser.add_argument('--forward_ndx', type=int, default=None)
        parser.add_argument("--domain", type=str, required=True)
        # # Demo
        # parser.add_argument('--process_id', type=int, default=None)  # for experimental mode
        # parser.add_argument('--m_target_index', type=int, default=4)  # for operation mode
        # parser.add_argument('--forward_ndx', type=int, default=20)  # for operation mode
        # parser.add_argument("--domain", type=str, default=None)
        args = parser.parse_args()
        args.domain = get_domain_on_CDSW_env(args.domain)
        args = scp.ScriptParameters(args.domain, args, job_id_int=args.process_id).update_args()

        # Basically, operation mode
        if args.process_id is None and not (args.m_target_index is None) and not (args.forward_ndx is None):
            base_dirs = ['./save/model/rllearn', './save/result', './save/model/rllearn/buffer_save',
                         './save/tensorlog/market_timing', './save/result/selected']
            target_name = RUNHEADER.target_id2name(args.m_target_index)
            forward_ndx = str(args.forward_ndx)
            for base_dir in base_dirs:
                fn_list = [it for it in os.listdir(base_dir) if target_name in it and 'T' + forward_ndx in it]
                for fn in fn_list:
                    try:
                        fn = base_dir + os.sep + fn
                        if platform.system() == 'Windows':
                            fn = fn.replace('\\', '/')

                        if './save/model/rllearn' in fn and './save/model/rllearn/buffer_save' not in fn:
                            if auto_clean_envs.check_model_pool(fn, target_name, forward_ndx):
                                print('Delete: {}'.format(fn))
                                shutil.rmtree(fn, ignore_errors=True)
                        else:
                            print('Delete: {}'.format(fn))
                            shutil.rmtree(fn, ignore_errors=True)
                    except FileNotFoundError:
                        pass
        # experimental mode
        elif not (args.process_id is None):
            bool_delete_result = False
            bool_copy_tensorlog = True
            process_ids = list()
            process_ids.append(args.process_id)

            # Auto Delete for given process id
            for process_id in process_ids:
                try:
                    json_location = recent_procedure('./agent_log/working_model_p', process_id, 'r')
                    json_location = './save/model/rllearn/' + json_location
                    dict_RUNHEADER = util.json2dict('{}/agent_parameter.json'.format(json_location))
                    # re-load
                    for key in dict_RUNHEADER.keys():
                        RUNHEADER.__dict__[key] = dict_RUNHEADER[key]

                    # auto_clean
                    m_name = RUNHEADER.m_offline_buffer_file
                    m_name = m_name.split('/')
                    for subdirs, dirs, files in os.walk(current_root):
                        if platform.system() == 'Windows':
                            subdir_split = subdirs.replace('\\', '/').split('/')
                        else:
                            subdir_split = subdirs.split('/')

                        if m_name[-1] == subdir_split[-1]:
                            if not 'buffer_save' in subdirs:
                                if 'model' in subdirs:
                                    for file in files:
                                        filepath = subdirs + os.sep + file
                                        if 'model' in filepath and '.pkl' in filepath:
                                            print('Delete: {}'.format(filepath))
                                            shutil.rmtree(filepath, ignore_errors=True)
                                else:
                                    print('Delete: {}'.format(subdirs))
                                    shutil.rmtree(subdirs, ignore_errors=True)

                    m_name = json_location.split('/')
                    for subdirs, dirs, files in os.walk(current_root):
                        if platform.system() == 'Windows':
                            subdir_split = subdirs.replace('\\', '/').split('/')
                        else:
                            subdir_split = subdirs.split('/')

                        if m_name[-1] == subdir_split[-1]:
                            if not ('buffer_save' in subdirs):
                                if 'result' in subdirs:
                                    if bool_delete_result:
                                        print('Delete: {}'.format(subdirs))
                                        shutil.rmtree(subdirs, ignore_errors=True)
                                    else:
                                        pass
                                elif 'tensorlog' in subdirs:
                                    if bool_copy_tensorlog:
                                        print('copy: {} to {}'.format(subdirs, './save/result/' + m_name[-1]))
                                        shutil.copytree(subdirs, './save/result/' + m_name[-1] + '/' + m_name[-1])
                                    else:
                                        pass
                                    print('Delete: {}'.format(subdirs))
                                    shutil.rmtree(subdirs, ignore_errors=True)
                                else:
                                    print('Delete: {}'.format(subdirs))
                                    shutil.rmtree(subdirs, ignore_errors=True)
                except FileNotFoundError:
                    pass
        else:
            assert False, 'Required parameters are not given [process_id | m_target_index and forward_ndx]'
    except Exception as e:
        print('\n{}'.format(e))
        exit(1)
