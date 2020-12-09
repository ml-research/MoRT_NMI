import pickle
from mort.funcs_mcm import BERTSentence, BERTSentenceSubspace
from tqdm import tqdm

#data_user_study = pickle.load(open('data/parsed_yes_no.p', 'rb'))
data_user_study = pickle.load(open('data/parsed_yes_no_BERTBias.p', 'rb'))
actions = list(data_user_study.keys())

"""emb = BERTSentenceSubspace()

for action in tqdm(actions):
    bias_with_action, _, _ = emb.bias(action)
    data_user_study[action].append(bias_with_action[0])
    #break

pickle.dump(data_user_study, open('data/parsed_yes_no_BERTBias.p', 'wb'))"""

import csv
csv_columns = ['Action', 'Yes', 'No', 'BERT bias']

csv_file = "data/parsed_yes_no_BERTBias.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        for action in actions:
            data_row = [action,
                        data_user_study[action][0],
                        data_user_study[action][1],
                        data_user_study[action][2]]
            writer.writerow(data_row)
except IOError:
    print("I/O error")