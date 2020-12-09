from mort.dataMoral import *
from mort.funcs_mcm import mcm_template_quests
import csv
from tqdm import tqdm
import os
import argparse

#python mort/experiments_mcm.py --data neg50 --model bertsentence

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--data', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--model', default=None, type=str,
                    help='model name', required=True)
parser.add_argument('--single_question', action='store_true')
parser.add_argument('--paper_plot', action='store_true')


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_path = None
    if len(args.model.split('_')) > 1:
        model_name = args.model.replace(args.model.split('_')[0] + "_", "")
        checkpoint_path = 'retrain_use/skip_thoughts/trained_models/train_' + model_name + '/'

    experimental_quests_ = experimental_quests_paper

    if args.data == 'atomic':
        data = get_actions_verbs()
    elif args.data == 'context':
        data = get_actions()
    elif args.data == 'allcorpora':
        data = get_actions_all_datasources()
    elif args.data == 'pos50':
        data = list([d[0] for d in dos_50])
    elif args.data == 'neg50':
        data = list([d[0] for d in donts_50])
    elif args.data == 'userstudy':
        with open("/home/patrick/repositories/MoRT/data/user_study/userStudy_yes_no.csv", "r") as f:
            actions_keys_ = [row.split(',')[0] for row in f]
            actions_keys_ = actions_keys_[1:]
            data = actions_keys_
    else:
        raise ValueError("word selection not found")
    res = list()
    n = 20
    for i in tqdm(range(int(len(data) // n)+1)):
        batch = data[i*n: i*n + n]
        res += mcm_template_quests(experimental_quests_, batch, args.model, checkpoint_path)
    assert len(res) == len(data)

    res.sort(key=lambda x: x[0])
    # save data to file
    filename = 'mort/plt_Corr/results/bias/{}Verbs_{}Embedding{}{}.csv'.format(args.data,
                                                                             args.model,
                                                                             "_single" if args.single_question else "",
                                                                             "_paper_plot" if args.paper_plot else "")
    ensure_dir(filename)
    with open(filename, mode='w') as result_file:
        result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for elem in res:
            if args.paper_plot:
                elem_to_write = [elem[1], "{0:.3f}".format(elem[0])]
            else:
                elem_to_write = [elem[0], elem[1]]

            result_writer.writerow(elem_to_write)
    print('****** file has been saved.. to', filename)