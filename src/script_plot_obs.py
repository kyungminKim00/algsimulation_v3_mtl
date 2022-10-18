import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

filename = './save/tf_record/index_forecasting/if_x0_20_y20_v0/if_v0_cv00_train.pkl'
save_img_dir = './save/tf_record/index_forecasting/if_x0_20_y20_v0'
with open(filename, 'rb') as fp:
    dataset = pickle.load(fp)
    fp.close()


def plot_typeA():
    n_type_obs, n_dates, n_index = np.array(dataset[0]['structure/predefined_observation_total']).shape
    n_sample = 10
    n_date = 40
    sample_index = np.random.randint(0, n_index, n_sample)
    for idx in range(n_date):
        for index_idx in sample_index:
            tt = np.array(dataset[idx]['structure/predefined_observation_total'])
            tt = np.transpose(tt, [2, 0, 1])
            img = tt[index_idx]

            plt_manager = plt.get_current_fig_manager()
            plt_manager.resize(1860, 980)
            plt.subplot(4, 1, 1)
            plt.imshow(img[0:5])
            plt.subplot(4, 1, 2)
            plt.imshow(img[5:10])
            plt.subplot(4, 1, 3)
            plt.imshow(img[10:15])
            plt.subplot(4, 1, 4)
            plt.imshow(img[15:20])
            plt.pause(0.1)
            target = '{}/Plot_OBS/Type_A/index_{}'.format(save_img_dir, index_idx)
            path = Path(target)
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig('{}/{}.jpeg'.format(target, idx), format='jpeg', dpi=600)
            plt.close()


def plot_typeB():
    n_type_obs, n_dates, n_index = np.array(dataset[0]['structure/predefined_observation_total']).shape
    canvas_row = 5
    canvas_col = 5
    n_sample = canvas_row*canvas_col
    n_date = 40
    sample_index = np.random.randint(0, n_index, n_sample)
    """Plot Type B,  obs_0
    """
    for idx in range(n_date):
        tt = np.array(dataset[idx]['structure/predefined_observation_total'])
        tt = np.transpose(tt, [2, 0, 1])
        plt_manager = plt.get_current_fig_manager()
        plt_manager.resize(1860, 980)
        for index_idx in range(len(sample_index)):
            img = tt[sample_index[index_idx]]
            plt.subplot(canvas_row, canvas_col, index_idx + 1)
            plt.imshow(img[0:5])
        plt.pause(0.1)
        target = '{}/Plot_OBS/Type_B/OBS_0'.format(save_img_dir)
        Path(target).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/{}.jpeg'.format(target, idx), format='jpeg', dpi=600)
        plt.close()

    """Plot Type B,  obs_1
    """
    for idx in range(n_date):
        tt = np.array(dataset[idx]['structure/predefined_observation_total'])
        tt = np.transpose(tt, [2, 0, 1])
        plt_manager = plt.get_current_fig_manager()
        plt_manager.resize(1860, 980)
        for index_idx in range(len(sample_index)):
            img = tt[sample_index[index_idx]]
            plt.subplot(canvas_row, canvas_col, index_idx + 1)
            plt.imshow(img[5:10])
        plt.pause(0.1)
        target = '{}/Plot_OBS/Type_B/OBS_1'.format(save_img_dir)
        Path(target).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/{}.jpeg'.format(target, idx), format='jpeg', dpi=600)
        plt.close()

    """Plot Type B,  obs_2
    """
    for idx in range(n_date):
        tt = np.array(dataset[idx]['structure/predefined_observation_total'])
        tt = np.transpose(tt, [2, 0, 1])
        plt_manager = plt.get_current_fig_manager()
        plt_manager.resize(1860, 980)
        for index_idx in range(len(sample_index)):
            img = tt[sample_index[index_idx]]
            plt.subplot(canvas_row, canvas_col, index_idx + 1)
            plt.imshow(img[10:15])
        plt.pause(0.1)
        target = '{}/Plot_OBS/Type_B/OBS_2'.format(save_img_dir)
        Path(target).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/{}.jpeg'.format(target, idx), format='jpeg', dpi=600)
        plt.close()

    """Plot Type B,  obs_3
    """
    for idx in range(n_date):
        tt = np.array(dataset[idx]['structure/predefined_observation_total'])
        tt = np.transpose(tt, [2, 0, 1])
        plt_manager = plt.get_current_fig_manager()
        plt_manager.resize(1860, 980)
        for index_idx in range(len(sample_index)):
            img = tt[sample_index[index_idx]]
            plt.subplot(canvas_row, canvas_col, index_idx + 1)
            plt.imshow(img[15:20])
        plt.pause(0.1)
        target = '{}/Plot_OBS/Type_B/OBS_3'.format(save_img_dir)
        Path(target).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/{}.jpeg'.format(target, idx), format='jpeg', dpi=600)
        plt.close()


if __name__ == '__main__':
    plot_typeA()
    plot_typeB()