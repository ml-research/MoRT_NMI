import argparse
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from datetime import datetime

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--sub_num', default=None, nargs='*',
                    help='subject number', required=True)


def csv_to_dict(file):
    templates = ['Is it exemplary to', 'Is it recommended to', 'Should I', 'Is it ok to', 'Is it desired to', 'Shall I',
                 'Is it good behaviour to', 'May I']

    print(file)
    df = pd.read_csv(file, usecols=['trialsMoral.thisIndex', 'condition', 'questions', 'keyYN.keys', 'keyYN.rt'])
    df = df[6:149].values.tolist()
    #print(df)
    vmin = 6
    vmax = 0
    results = dict()
    for row in df:
        sent = row[0]
        if row[3] == 'left':
            answer = 1
        else:
            answer = -1

        if np.isnan(row[4]):
            continue

        if row[4] < vmin:
            vmin = row[4]
        if row[4] > vmax:
            vmax = row[4]

        for temp in templates:
            if temp in sent:
                action = sent.replace(temp, ' ')
                action = action.replace('?', ' ')
                action = action.strip()
                #print(action)
                if not(action in results.keys()):
                    results[action] = [answer, row[4]]

    print(results)
    # print(vmin, vmax)

    # moral_score = dict()
    # for key in results.keys():
    #     moral_score[key] = results[key][0] * (1 - ((results[key][1] - vmin) / (vmax - vmin)))
    #
    # print(moral_score)
    return results


    #sub_data = df[['questions', 'keyYN.keys', 'keyYN.rt']][6:149]
    # print(sub_data)

    #data_list = sub_data.values.tolist()

    # data_dict = dict()
    # for trial in data_list:
    #     data_dict[int(trial[0])] = trial[1:]
    # return data_dict


if __name__ == "__main__":
    args = parser.parse_args()
    counts_dict = dict()

    # sub_file = glob.glob('./data/Behavioral/MCM_psychopy/{}_*.csv'.format(sub_num))
    sub_files = glob.glob('./data/Behavioral/MCM_psychopy/0*.csv')

    for sub_file in sub_files:
        #print(sub_file)
        sub_dict = csv_to_dict(sub_file)

        for key in sub_dict.keys():
            ans = sub_dict[key][0]
            if key in counts_dict.keys():
                if ans == 1:
                    counts_dict[key][0] += 1
                else:
                    counts_dict[key][1] +=1
            else:
                if ans == 1:
                    counts_dict[key] = [1, 0]
                else:
                    counts_dict[key] = [0, 1]

    print(counts_dict)
    import pickle
    pickle.dump(counts_dict, open('./data/Behavioral/parsed_yes_no.p', 'wb'))

    action = list()
    yes = list()
    no = list()
    for key in counts_dict.keys():
        action += [key]
        yes += [counts_dict[key][0]]
        no += [counts_dict[key][1]]

    # print(yes)
    width = 0.5

    row = 6
    col = 2
    total = 0
    fig, axes = plt.subplots(col, row, figsize=(32,16))
    #
    sns.set(style='ticks', palette='Set2')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=False)

    print(len(yes))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        yes_ = yes[total*10 : total*10 + 10]
        no_ = no[total * 10: total * 10 + 10]
        action_ = action[total * 10: total * 10 + 10]
        ind = range(0, len(yes_))
        p1 = ax.bar(ind, yes_, width)
        p2 = ax.bar(ind, no_, width,
                     bottom=yes_)
        ax.set_xticks(ind)
        ax.set_xticklabels(action_, rotation='vertical')

        total += 1
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        # ax.set_xlabel("nm", labelpad=10, fontsize=34, color="#444444")

    plt.subplots_adjust(top=0.92, bottom=0.28, left=0.10, right=0.95, hspace=1.25,
                        wspace=0.35)
    plt.show()