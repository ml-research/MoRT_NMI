import csv
import os

fname = "./data/user_study/userStudy_yes_no_regional.csv"
fname_out = "./data_/correlation/userStudy_scores_regional.csv"
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
os.makedirs(os.path.dirname(fname_out), exist_ok=True)
with open(fname_out, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)
    for action in actions:
        data_row = [
            action,
            bias_dict[action]
        ]
        writer.writerow(data_row)
