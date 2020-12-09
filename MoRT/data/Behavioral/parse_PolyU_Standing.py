##############
#   Cigdem TURAN
#   23 August 2019
#   Darmstadt
#
#   Parsing code for PolyU Standing Study
#   PsychoPy and OpenSesame
#
#
# HOW TO RUN:
#   Requirements:
#       pandas: run in the terminal: pip install pandas
#
#   Execute:
#       --path_csv: path to the folder that has csv output
#       --path_data_folder: path to the folder that has all subject data
#       --sub_num: number of the subject to be executed
#
#   Run this on the terminal:
#   python parse_PolyU_Standing.py --path_csv /path/to/csv/ --path_data_folder /path/to/data/folder/ --sub_num 132
#       or if you wanna run multiple subjects at the same time:
#   python parse_PolyU_Standing.py --path_csv /path/to/csv/ --path_data_folder /path/to/data/folder/ --sub_num 132 133 134
#
#   Path example: /Users/ml-cturan/Projects/PolyU_Standing/
#
#   ATTENTION:
#       The paths needs to finish with / as seen in the example
##############

import argparse
import pandas as pd
import glob
from datetime import datetime


parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--path_csv', default=None, type=str,
                    help='path to csv file', required=True)
parser.add_argument('--path_data_folder', default=None, type=str,
                    help='path to subjects', required=True)
parser.add_argument('--sub_num', default=None, nargs='*',
                    help='subject number', required=True)


def csv_to_dict(file):
    print(file)
    df = pd.read_csv(file)
    sub_data = df[['count_showImage', 'mainStim', 'avg_rt', 'imgBeg', 'imgEnd']][5:]
    # print(sub_data)

    data_list = sub_data.values.tolist()

    data_dict = dict()
    for trial in data_list:
        data_dict[int(trial[0])] = trial[1:]
    return data_dict


def str_to_time(str_time):
    # print(str_time)
    if len(str_time.split('_')) == 3:
        str_time += '_00000'
    time_datetime = datetime.strptime(str_time, '%H_%M_%S_%f')
    return time_datetime


def line_to_time_joint(str_line):
    time_point = str_line.strip('\n').split(', ')[0]
    # print(time_point)
    if time_point is '':
        return [], []
    time_datetime = str_to_time(time_point[:-1])
    str_line = str_line.replace('-∞', '-9999')
    joints_data = str_line.split(', ')[1:-1]
    joints_data = list(map(float, joints_data))
    return time_datetime, joints_data


def create_joints_dict(joints_data):
    joints_dict = dict()
    if len(joints_data) == 50:
        for i in range(25):
            joints_dict['joint_{}x'.format(i + 1)] = joints_data[i * 2]
            joints_dict['joint_{}y'.format(i + 1)] = joints_data[i * 2 + 1]
    elif len(joints_data) == 75:
        for i in range(25):
            joints_dict['joint_{}x'.format(i + 1)] = joints_data[i * 3]
            joints_dict['joint_{}y'.format(i + 1)] = joints_data[i * 3 + 1]
            joints_dict['joint_{}z'.format(i + 1)] = joints_data[i * 3 + 2]
    return joints_dict


def run_for_sub(sub_num, path_data_folder, trial_dict, num_dim):
    data_list = list()

    trial_num = 0
    image_name = trial_dict[trial_num][0]
    rt = trial_dict[trial_num][1]
    time_aim = str_to_time(trial_dict[trial_num][2][:-1])
    flag_save = False
    flag_end = False

    len_files = len(glob.glob('{}{}/joints/joints{}_*.txt'.format(path_data_folder, sub_num, num_dim)))

    for cnt_2D in range(len_files):
        file_2D = glob.glob('{}{}/joints/joints{}_{}_*.txt'.format(path_data_folder, sub_num, num_dim, cnt_2D))
        print(file_2D[0])
        fp = open(file_2D[0], 'r')
        lines = fp.readlines()
        for line in lines:
            time_frame, joints_val = line_to_time_joint(line)
            #print(time_frame)
            if not joints_val:
                continue
            # print(time_ins)
            if not flag_save:
                if time_frame < time_aim:
                    continue
                else:
                    print(joints_val)
                    flag_save = True
                    row_dict = {**{'trial_num': trial_num, 'image_name': image_name, 'rt': rt,
                                   'time_point': datetime.strftime(time_frame, '%H_%M_%S_%f')},
                                **create_joints_dict(joints_val)}
                    data_list.append(row_dict)
                    # print(row_dict)
                    time_aim = str_to_time(trial_dict[trial_num + 1][2][:-1])
            else:
                if time_frame < time_aim:
                    row_dict = {**{'trial_num': trial_num, 'image_name': image_name, 'rt': rt,
                                   'time_point': datetime.strftime(time_frame, '%H_%M_%S_%f')},
                                **create_joints_dict(joints_val)}
                    data_list.append(row_dict)
                    # print(row_dict)
                else:
                    try:
                        trial_num += 1
                        image_name = trial_dict[trial_num][0]
                        rt = trial_dict[trial_num][1]
                        time_aim = str_to_time(trial_dict[trial_num + 1][2][:-1])
                        row_dict = {**{'trial_num': trial_num, 'image_name': image_name, 'rt': rt,
                                       'time_point': datetime.strftime(time_frame, '%H_%M_%S_%f')},
                                    **create_joints_dict(joints_val)}
                        data_list.append(row_dict)
                    except KeyError:
                        flag_end = True
                        break
        fp.close()
        if flag_end:
            break

    data_df = pd.DataFrame(data_list)

    trial_file = './subject-{}_joints{}.csv'.format(sub_num, num_dim)
    data_df.to_csv(trial_file, index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    path_data_folder = args.path_data_folder

    for sub_num in args.sub_num:
        sub_num = int(sub_num)
        csv_file = '{}subject-{}.csv'.format(args.path_csv, sub_num)
        trial_dict = csv_to_dict(csv_file)

        run_for_sub(sub_num, path_data_folder, trial_dict, '2D')
        run_for_sub(sub_num, path_data_folder, trial_dict, '3D')
