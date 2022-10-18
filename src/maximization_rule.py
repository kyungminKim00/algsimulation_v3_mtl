import pandas as pd
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from util import write_pickle, read_pickle
import os
import argparse


def get_data(files=None):
    data = list()
    for idx in range(len(files)):
        data.append(pd.read_csv(files[idx]))
    
    for idx in range(len(data)):
        if idx == 0:
            i_data = data[idx]
        else:
            j_data = data[idx]
            if idx == 1:
                tot = pd.merge(i_data, j_data, left_on="cnt", right_on="cnt", how="outer")
            else:
                tot = pd.merge(tot, j_data, left_on="cnt", right_on="cnt", how="outer")
        
    is_idx_1 = tot["expected_label_x"] == 2
    is_idx_2 = tot["expected_label_y"] == 2
    is_idx_3 = tot["expected_label"] == 2

    tot_refined = tot[~(is_idx_1 & is_idx_2 & is_idx_3)]
    tot_refined = tot_refined[
        [
            "cnt",
            "expected_label_x",
            "expected_label_y",
            "expected_label",
            "expected_return_x",
            "expected_return_y",
            "expected_return",
            "real_return",
        ]
    ]

    for it in ["expected_label_x", "expected_label_y", "expected_label"]:
        tot_refined[it].fillna(2, inplace=True)

    for it in ["expected_return_x", "expected_return_y", "expected_return"]:
        tot_refined[it].fillna(0, inplace=True)

    pd.DataFrame(data=tot_refined).to_csv(
        "{}/{}_total_historical_return.csv".format(root_dir, target_name), index=None
    )
    return tot_refined


def search_full_combination(classes, num_predictor, tot_refined):
    allowed_rules = list()
    for i in classes:
        for j in classes:
            for k in classes:
                is_idx_1 = tot_refined["expected_label_x"] == i
                is_idx_2 = tot_refined["expected_label_y"] == j
                is_idx_3 = tot_refined["expected_label"] == k
                m_data = tot_refined[is_idx_1 & is_idx_2 & is_idx_3]

                a = np.sum(m_data["expected_return_x"])
                b = np.sum(m_data["expected_return_y"])
                c = np.sum(m_data["expected_return"])

                print(
                        "Permutations[{}]: {}-{}-{}: {}, {}, {}".format(
                            m_data.shape[0],
                            i,
                            j,
                            k,
                            a,
                            b,
                            c,
                        )
                    )
                
                v_max = np.max([a,b,c])
                if v_max == 0:
                    predictor = None
                    expected_label = 2
                else:
                    predictor = np.argwhere(np.array([a,b,c]) == v_max)[0].tolist()[0]
                    if predictor == 0:
                        predictor = 'expected_label_x'
                        predictor_return = 'expected_return_x'
                    elif predictor == 1:
                        predictor = 'expected_label_y'
                        predictor_return = 'expected_return_y'
                    elif predictor == 2:
                        predictor = 'expected_label'
                        predictor_return = 'expected_return'
                allowed_rules.append(['{}-{}-{}'.format(i, j, k), [predictor, predictor_return]])

    return dict(allowed_rules)


def total_return_chart_plot(plt, total_return_chart_data):
    tmp_data = np.array(total_return_chart_data)
    realized_y = np.array(tmp_data[:, 0], dtype=np.float).tolist()
    realized_y = np.cumsum(realized_y)
    expected_y = np.array(tmp_data[:, 1], dtype=np.float).tolist()
    expected_y = np.cumsum(expected_y)
    dates = np.array(tmp_data[:, 2]).tolist()

    (fig, ax) = plt.subplots()
    ax.plot(
        dates, 
        realized_y, 
        color="black",
        label="realized_y")
    plt.xticks(np.arange(0, len(dates), 1000))
    
    ax.plot(
        dates,
        expected_y,
        color="red",
        label="expected_y",
    )

    # Save graph to file.
    plt.title(
        "Return:{}, (Exptect) Return:{}".format(
            round(np.sum(realized_y[-1]), 3),
            round(np.sum(expected_y[-1]), 3),
        )
    )
    plt.legend(loc="best")

    img_name = "rule_applied.png"
    plt.savefig("{}/{}_{}".format(root_dir, target_name, img_name))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--target", type=str, default='FTSE')
    parser.add_argument("--g_combination", type=int, default=0)
    args = parser.parse_args()

    global target_name, root_dir

    target_name = args.target
    root_dir = './temp'
    target_dir = ['/'.join([root_dir, item]) for item in os.listdir(root_dir) if target_name in item]
    files = list()
    for it in target_dir:
        if os.path.isdir(it):
            for item in os.listdir(it):
                if '.csv' in item:
                    files.append('/'.join([it, item]))
        
    tot_refined = get_data(files=files)
    if args.g_combination:
        allowed_rules = read_pickle('/'.join([root_dir, 'g_allowed_rules.pkl']))
    else:
        allowed_rules = search_full_combination(classes=[0, 1, 2], num_predictor=3, tot_refined=tot_refined)
        write_pickle(allowed_rules, '{}/{}_allowed_rules.pkl'.format(root_dir, target_name))
        

    voted_result_with_restriction = list()
    col_name = ['real_return', 'expected_return', 'cnt']
    for index, row in tot_refined.iterrows():
        real_return = row['real_return']
        cnt = row['cnt']

        r_combination = '{}-{}-{}'.format(int(row['expected_label_x']), int(row['expected_label_y']), int(row['expected_label']))
        predictor, predictor_return = allowed_rules[r_combination]
        if predictor is None:
            expected_return = 0
            expected_label = 2
        else:
            expected_label = row[predictor]
            expected_return = row[predictor_return]
        
        voted_result_with_restriction.append([real_return, expected_return, cnt])
    
    total_return_chart_plot(plt, voted_result_with_restriction)
