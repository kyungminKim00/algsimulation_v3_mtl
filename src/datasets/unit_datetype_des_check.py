import header.index_forecasting.RUNHEADER as RUNHEADER
from util import get_unique_list

import pandas as pd
from collections import OrderedDict
import numpy as np
import datetime


def add_item(it, d_f_summary, var_list, b_current_pt_only=False, b_percent_except=False, b_use_all=False):
    Condition = d_f_summary['var_name'] == it
    selected = d_f_summary[Condition]
    if b_current_pt_only:
        if selected['units'].values[0] in ['currency', 'pt'] and \
                (selected['data_type'] == 'Daily').values[0]:
            var_list.append(it)
    if b_percent_except:
        if (selected['abbreviation'].values[0] in ['cnvxt']) or (selected['abbreviation'].values[0] in ['bid_yld']):
            pass
        else:
            var_list.append(it)
    if b_use_all:
        var_list.append(it)

    return var_list

def type_check(d_f_summary, t1, t2):
    selected_item = d_f_summary['var_name']
    T1 = selected_item == t1
    T2 = selected_item == t2
    conditions = ['units', 'category', 'data_type', 'abbreviation']
    for cond in conditions:
        if d_f_summary[T1][cond].values[0] == d_f_summary[T2][cond].values[0]:
            pass
        else:
            return False
    return True


def quantising_vars(data, ids_to_var_names):
    desc = pd.read_csv(RUNHEADER.var_desc)
    categories = list(desc['category'])
    quantise = list()
    for it in list(set(desc['category'])):
        quantise.append([it ,int(categories.count(it) * 0.2)])
    
    num_max_vars = OrderedDict(quantise)
    new_ids_to_var_names = list()
    duplicate_idx = list()
    for key, max_val in num_max_vars.items():
        cnt = 0
        for ids, ids_name in ids_to_var_names.items():
            if duplicate_idx.count(ids) == 0:
                if '-' in ids_name:
                    new_ids_to_var_names.append([int(ids), ids_name])
                    duplicate_idx.append(ids)
                else:
                    t_key = desc[desc['var_name'] == ids_name]['category'].tolist()[0]
                    if (cnt <= max_val) and (key == t_key):
                        new_ids_to_var_names.append([int(ids), ids_name])
                        cnt = cnt + 1
                        duplicate_idx.append(ids)
    new_ids_to_var_names = sorted(new_ids_to_var_names, key=lambda aa: aa[0])
    selected_idxs = np.array(new_ids_to_var_names, dtype=np.object)[:, 0].tolist()
    
    # update
    data = data[:, selected_idxs]
    ids_to_var_names = OrderedDict(new_ids_to_var_names)
    return data, ids_to_var_names

def script_run(f_name=None):
    if f_name is None:  # Demo Test
        f_index = './datasets/rawdata/index_data/data_vars_US10YT_Indices.csv'
        max_x = 300
        assert False, 'Demo Disabled'
    else:  # header configured
        f_index = f_name
        max_x = RUNHEADER.max_x
        print('script_run - f_index: {}'.format(f_index))
        print('script_run - max_x: {}'.format(max_x))
    f_summary = RUNHEADER.var_desc

    # load data
    d_f_index = pd.read_csv(f_index, header=None).values.squeeze()
    d_f_summary = pd.read_csv(f_summary)

    # get variables except derived variables
    b_use_derived_vars = False
    var_list = list()
    for it in d_f_index:
        if '-' in it:
            if b_use_derived_vars:
                # Not in use -
                a, b = it.split('-')
                var_list = add_item(a, d_f_summary, var_list, b_percent_except=True)
                var_list = add_item(b, d_f_summary, var_list, b_percent_except=True)
            else:
                pass
        else:
            var_list = add_item(it, d_f_summary, var_list, b_percent_except=True)

    # merge & save
    source_1_head = get_unique_list(var_list)[:int(max_x*0.5)]  # none derived vars
    source_2_head = get_unique_list(d_f_index)[:int(max_x*0.5)]
    source_1_tail = get_unique_list(d_f_index)[int(max_x * 0.5):]
    

    my_final_list = OrderedDict.fromkeys(source_1_head + source_2_head)
    my_final_list = list(my_final_list) + source_1_tail
    pd.DataFrame(data=my_final_list, columns=['VarName']). \
        to_csv(f_index, index=None, header=None)
    print('{} has been saved'.format(f_index))

    # save desc
    basename = f_index.split('.csv')[0]
    write_var_desc(my_final_list, d_f_summary, basename)

    # var_desc = list()
    # for it in my_final_list:
    #     if '-' in it:
    #         for cnt in range(2):
    #             Condition = d_f_summary['var_name'] == it.split('-')[cnt]
    #             tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
    #             var_desc.append(tmp)
    #     else:
    #         Condition = d_f_summary['var_name'] == it
    #         tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
    #         var_desc.append(tmp)
    # pd.DataFrame(data=var_desc, columns=d_f_summary.keys()[1:]). \
    #     to_csv(basename + '_desc.csv')
    # print('{} has been saved'.format(f_index.split('.csv')[0] + '_desc.csv'))


def write_var_desc(my_final_list, d_f_summary, basename):
    # save desc
    var_desc = list()
    for it in my_final_list:
        if '-' in it:
            for cnt in range(2):
                Condition = d_f_summary['var_name'] == it.split('-')[cnt]
                tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
                var_desc.append(tmp)
        else:
            Condition = d_f_summary['var_name'] == it
            tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
            var_desc.append(tmp)
    pd.DataFrame(data=var_desc, columns=d_f_summary.keys()[1:]). \
        to_csv(basename + '_desc.csv')
    print('{} has been saved'.format(basename + '_desc.csv'))

def write_var_desc_with_correlation(my_final_list, my_final_cov, d_f_summary, basename, performed_date):
    time_now = (
            str(datetime.datetime.now())[:-16]
            .replace(":", "-")
            .replace("-", "")
            .replace(" ", "_")
        )
    time_now = performed_date
    
    # save desc
    var_desc = list()
    for idx in range(len(my_final_list)):
        it = my_final_list[idx]
        if '-' in it:
            pass
        else:
            Condition = d_f_summary['var_name'] == it
            tmp = [time_now, basename.split('/')[-2].split('_')[-2][1:], RUNHEADER.target_name] + d_f_summary[Condition].values.squeeze().tolist()[1:] + [my_final_cov[-1, idx]]
            var_desc.append(tmp)
    pd.DataFrame(data=var_desc, columns=['performed_date', 'forward', 'mrkt_cd'] + list(d_f_summary.keys()[1:]) + ['score']). \
        to_csv(basename, index=None, sep='|')
    print('{} has been saved'.format(basename))


if __name__ == '__main__':
    script_run()







