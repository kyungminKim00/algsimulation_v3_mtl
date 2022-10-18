import pickle

from contextlib import contextmanager
import time
import argparse

# time check
@contextmanager
def funTime(func_name):
    start = time.clock()
    yield
    end = time.clock()
    interval = end - start
    print('\n== Time cost for [{0}] : {1}'.format(func_name, interval))


def load(filepath, method):
    with open(filepath, 'rb') as fs:
        if method == 'pickle':
            data = pickle.load(fs)
    fs.close()
    return data

@funTime('save test')
def save(filepath, method, obj):
    with open(filepath, 'wb') as fs:
        if method == 'pickle':
            pickle.dump(obj, fs)
    fs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    # init args
    parser.add_argument('--dataset_version', type=str, default='v15')  # save as v7 'v7'
    parser.add_argument('--forward_ndx', type=int, default=20)
    args = parser.parse_args()

    forward_ndx = str(args.forward_ndx)
    dataset_version = str(args.dataset_version)

    filepath_train = '../save/tf_record/index_forecasting/if_x0_20_y{}_{}/if_{}_cv00_train.pkl'.format(forward_ndx, dataset_version, dataset_version)
    filepath_validation = '../save/tf_record/index_forecasting/if_x0_20_y{}_{}/if_{}_cv00_validation.pkl'.format(forward_ndx, dataset_version, dataset_version)
    filepath_test = '../save/tf_record/index_forecasting/if_x0_20_y{}_{}/if_{}_cv00_test.pkl'.format(forward_ndx, dataset_version, dataset_version)
    meta = '../save/tf_record/index_forecasting/if_x0_20_y{}_{}/meta'.format(forward_ndx, dataset_version)
    filepath_test = [filepath_train, filepath_validation, filepath_test]

    # print out
    meta_info = load(meta, 'pickle')
    print('=== meta_info ===')
    print('x_seq: {}'.format(meta_info['x_seq']))
    print('forecast: {}'.format(meta_info['forecast']))
    # print('test_set_start: {}'.format(meta_info['test_set_start']))
    # print('test_set_end: {}'.format(meta_info['test_set_end']))

    print('\n=== Data Set ===')
    for fn in filepath_test:
        data = load(fn, 'pickle')
        target_column = [[it['date/base_date_label'], it['date/prediction_date_label']] for it in data]
        string_print = target_column[-1]
        print('[{}] Base: {}, Prediction: {}'.format(fn.split('_')[-1], string_print[0], string_print[1]))


