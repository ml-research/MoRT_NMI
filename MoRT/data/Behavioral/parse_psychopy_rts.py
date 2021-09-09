import argparse
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import csv


def find_action_template(subfile):
    print('here')
    templates = ['Is it exemplary to', 'Is it recommended to', 'Should I', 'Is it ok to', 'Is it desired to', 'Shall I',
                 'Is it good behaviour to', 'May I', 'Is it polite to']
    list_atomic = ['acknowledge', 'apologize', 'blame', 'borrow', 'compliment', 'cuddle', 'drink', 'eat', 'greet',
                   'harm', 'have a gun', 'help', 'kill', 'lie', 'love', 'misinform', 'pursue', 'smile', 'steal',
                   'talk', 'torture', 'travel', 'waste']

    df = pd.read_csv(subfile, usecols=['questions'])
    df = df[6:149].values.tolist()
    results = dict()
    for row in df:
        sent = row[0]
        for temp in templates:
            if temp in sent:
                action = sent.replace(temp, ' ')
                action = action.replace('?', ' ')
                action = action.strip()
                if not(action in results.keys()):
                    results[action] = temp

    print(results)

    with open('./data/Behavioral/userStudy_action_template_regional.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["action", "question"])
        for key in results.keys():
            writer.writerow([key, results[key]])



def csv_to_list(file):
    templates = ['Is it exemplary to', 'Is it recommended to', 'Should I', 'Is it ok to', 'Is it desired to', 'Shall I',
                 'Is it good behaviour to', 'May I', 'Is it polite to', "Is it okto"]
    list_atomic = ['acknowledge', 'apologize', 'blame', 'borrow', 'compliment', 'cuddle', 'drink', 'eat', 'greet',
                   'harm', 'have a gun', 'help', 'kill', 'lie', 'love', 'misinform', 'pursue', 'smile', 'steal',
                   'talk', 'torture', 'travel', 'waste']

    print(file)
    sub_num = file.split('/')[4].split('_')[0]
    count = 0
    df = pd.read_csv(file, usecols=['trialsMoral.thisIndex', 'questions', 'keyYN.keys', 'keyYN.rt'])
    df = df[6:149].values.tolist()
    results = list()
    for row in df:
        row.append(int(sub_num))
        sent = row[0]

        if np.isnan(row[3]):
            count += 1
            row[3] = 5.0
            #print(row)

        if row[2] == 'left':
            row[2] = 'yes'
        elif row[2] == 'right':
            row[2] = 'no'
        else:
            row[2] = 'none'

        if "Is it exemplary pursue" in sent:
            row[0] = "pursue money"
            results.append(row)
        else:
            for temp in templates:
                if temp in sent:
                    action = sent.replace(temp, ' ')
                    action = action.replace('?', ' ')
                    action = action.strip()
                    #print(action)

                    if 'have a gun' in action:
                        if sent == 'Is it recommended to have a gun?':
                            continue
                        else:
                            row[0] = action
                            results.append(row)
                            #print(row)

                    else:
                        if action.split(' ')[0] in list_atomic:
                            row[0] = action
                            results.append(row)
                            #print(row)

    results_diff = list()
    flag_diff = 0
    for row in results:
        if row[0] in list_atomic:
            if flag_diff == 0:
                rt_first = row[3]
                flag_diff = 1
            else:
                rt_second = row[3]
                row[3] = rt_second - rt_first ## before it was rt_first - rt_second
                results_diff.append(row)
                print(row)
                flag_diff = 0
        else:
            if flag_diff == 1:
                row[3] = row[3] - rt_first
            results_diff.append(row)
            print(row)
    #print(count)
    return results_diff


def process_bias(sub_all):
    list_atomic = ['acknowledge', 'apologize', 'blame', 'borrow', 'compliment', 'cuddle', 'drink', 'eat', 'greet',
                   'harm', 'have a gun', 'help', 'kill', 'lie', 'love', 'misinform', 'pursue', 'smile', 'steal',
                   'talk', 'torture', 'travel', 'waste']
    bias_dict = dict()
    for row in sub_all:
        action = row[0]
        if action in bias_dict.keys():
            if row[2] == 'yes':
                bias_dict[action][0] +=1
            elif row[2] == 'no':
                bias_dict[action][1] += 1
        else:
            if row[2] == 'yes':
                bias_dict[action] = [1, 0]
            elif row[2] == 'no':
                bias_dict[action] = [0, 1]

    for key in bias_dict.keys():
        votes = bias_dict[key]
        bias = votes[0] / (votes[0] + votes[1])
        #print(key, bias)
        bias_dict[key] = bias


    atomic_bias_dict = dict()
    for key in bias_dict.keys():
        if key in list_atomic:
            if bias_dict[key] < 0.5:
                atomic_bias_dict[key] = [bias_dict[key], 'neg']
            else:
                atomic_bias_dict[key] = [bias_dict[key], 'pos']

    for act in list_atomic:
        bias_dict.pop(act)

    bias_diff_key = dict()
    for key in bias_dict.keys():
        if key.split(' ')[0] == 'have':
            bias_diff = bias_dict[key] - atomic_bias_dict['have a gun'][0]
            bias_list = [bias_diff, atomic_bias_dict['have a gun'][1]]
        else:
            bias_diff = bias_dict[key] - atomic_bias_dict[key.split(' ')[0]][0]
            bias_list = [bias_diff, atomic_bias_dict[key.split(' ')[0]][1]]

        bias_diff_key[key] = bias_list
        print(key, bias_list)

    # for key in bias_dict.keys():
    #     bias_dict[key] = [bias_dict[key]] + bias_diff_key[key]
    #     print(bias_dict[key])
    return bias_diff_key


def prepare_csv(sub_all, bias_diff_dict):
    list_atomic = ['acknowledge', 'apologize', 'blame', 'borrow', 'compliment', 'cuddle', 'drink', 'eat', 'greet',
                   'harm', 'have a gun', 'help', 'kill', 'lie', 'love', 'misinform', 'pursue', 'smile', 'steal',
                   'talk', 'torture', 'travel', 'waste']

    item_list = list()
    for item in bias_diff_dict.keys():
        item_list.append(item)

    item_list.sort()
    item_dict = dict()
    for i, item in enumerate(item_list):
        item_dict[item] = i
        print(item, i)

    list_atomic.sort()
    list_atomic_dict = dict()
    for i, item in enumerate(list_atomic):
        list_atomic_dict[item] = i

    diff_rt_dict = {}
    for i in range(1,30):
        diff_rt_dict[i] = {}
    for row in sub_all:
        if row[0] in list_atomic:
            diff_rt_dict[row[4]][row[0]] = row[3]

    count_ts_inside = 0
    count_pre = 0
    with open('./data/Behavioral/userStudy_analysis_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["subid", "itemid", "item", "answer", "ts", "ts_in", "diff_rt", "rt_action", "bias_atomic", "diff_bias", "blockid"])
        for row in sub_all:
            #print(row)
            if row[0] in list_atomic:
                continue
            if row[0].split(' ')[0] == 'have':
                item_atomic = 'have a gun'
            else:
                item_atomic = row[0].split(' ')[0]

            if count_pre == 0:
                count_ts_inside = 1
            else:
                if int(row[1]) - count_pre != 1:
                    count_ts_inside = 1
                else:
                    count_ts_inside += 1


            #print(row[4], item_dict[row[0]], row[0], int(row[1]), diff_rt_dict[row[4]][item_atomic], row[3],
                  #bias_diff_dict[row[0]][1], bias_diff_dict[row[0]][0])
            writer.writerow([row[4], item_dict[row[0]], row[0], row[2], int(row[1]), count_ts_inside, diff_rt_dict[row[4]][item_atomic], row[3],
                            bias_diff_dict[row[0]][1], bias_diff_dict[row[0]][0], list_atomic_dict[item_atomic]])
            #print(count_ts_inside)
            count_pre = int(row[1])


if __name__ == "__main__":

    sub_files = glob.glob('./data/Behavioral/MCM_psychopy/0*.csv')
    count_sub = 0
    sub_all = list()
    for sub_file in sub_files:
        # print(sub_file)
        count_sub += 1
        sub_all.extend(csv_to_list(sub_file)[:])
    print(sub_all)
    #print(count_sub)

    bias_diff_dict = process_bias(sub_all)
    prepare_csv(sub_all, bias_diff_dict)

    1 / 0

    sub_num = '004'
    sub_file = glob.glob('./data/Behavioral/MCM_psychopy/{}_*.csv'.format(sub_num))

    find_action_template(sub_file[0])


