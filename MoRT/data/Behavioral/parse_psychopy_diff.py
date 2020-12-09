import argparse
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import csv


def calculate_series(block_dict):
    with open('./data/Behavioral/userStudy_analysis_cum_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["subid", "blockid", "diff_rt", "rt_mean", "rt_mean_abs", "rt_ratio", "bias_mean", "bias_mean_abs", "bias_ratio"])

        diff_cum_list = list()
        for key in block_dict.keys():
            ###print(key)
            block_data = block_dict[key]
            #print(block_data)
            sub_dict = dict()
            for row in block_data:
                ###print(row[0])
                if row[0] not in sub_dict.keys():
                    sub_dict[row[0]] = list()

                sub_dict[row[0]].append(row)

            for key_sub in sub_dict.keys():
                sub_data = sub_dict[key_sub]
                ###print(sub_data)

                rt_mean = 0
                rt_mean_abs = 0
                rt_pos = 0
                rt_neg = 0
                bias_mean = 0
                bias_mean_abs = 0
                bias_pos = 0
                bias_neg = 0
                k = 0
                flag_nosave = 0
                for row_sub in sub_data:
                    rt_sub = float(row_sub[7])
                    answer_trial = row_sub[3]
                    if answer_trial == "none":
                        flag_nosave = 1
                        print("hello")
                    rt_mean += rt_sub
                    rt_mean_abs += abs(rt_sub)
                    if rt_sub < 0:
                        rt_neg += 1
                    else:
                        rt_pos += 1

                    bias_sub = float(row_sub[9])
                    bias_mean += bias_sub
                    bias_mean_abs += abs(bias_sub)
                    if bias_sub < 0:
                        bias_neg += 1
                    else:
                        bias_pos += 1

                    k += 1

                if rt_neg == 0:
                    rt_ratio = 1
                else:
                    rt_ratio = rt_pos/rt_neg
                    rt_ratio /= k
                rt_mean /= k
                rt_mean_abs /= k
                rt_pos /= k
                rt_neg /= k
                if bias_neg == 0:
                    bias_ratio = 1
                else:
                    bias_ratio = bias_pos/bias_neg
                    bias_ratio /= k
                bias_mean /= k
                bias_mean_abs /= k
                bias_pos /= k
                bias_neg /= k

                ##print(rt_mean)
                ##print(rt_mean_abs)
                ##print(rt_pos, rt_neg, rt_ratio)

                if flag_nosave == 1:
                    continue

                writer.writerow([key_sub, key, row_sub[6], rt_mean, rt_mean_abs, rt_ratio, bias_mean, bias_mean_abs,
                                bias_ratio])
                #break


        #print(sub_dict.keys())
        #print(sub_dict["1"])


if __name__ == "__main__":
    block_dict = dict()
    with open('./data/Behavioral/userStudy_analysis_3.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=',')
        for row in csv_reader:
            if row[0] == "subid":
                ##print("hello")
                continue

            if row[10] not in block_dict.keys():
                block_dict[row[10]] = list()

            block_dict[row[10]].append(row)


    #print(block_dict["1"])

    calculate_series(block_dict)

    # for key in block_dict.keys():
    #     print(type(key))
    #
    # print(len(block_dict.keys()))