import numpy as np
import csv

fname = "/home/patrick/repositories/MoRT/data/user_study/userStudy_yes_no.csv"
fname_out = "/home/patrick/repositories/MoRT/data/correlation/userStudy_scores.csv"
sentences_ = list()

actions = list()

bias_dict = dict()
with open(fname, "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        line = line.rstrip('\n')
        vect = line.split(',')
        actions.append(vect[0])
        acc = float(vect[1]) / (float(vect[1]) + float(vect[2]))
        bias_dict[vect[0]] = acc

csv_columns = ['Action', 'Score']
with open(fname_out, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)
    for action in actions:
        data_row = [
            action,
            bias_dict[action]
        ]
        writer.writerow(data_row)
