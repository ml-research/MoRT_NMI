from mort.funcs_mcm import BERTSentence, BERTSentenceSubspace
from tqdm import tqdm
import csv
from mort.dataMoral import dos_50, donts_50
import os

actions_keys_dos = list([d[0] for d in dos_50])
actions_keys_donts = list([d[0] for d in donts_50])

question = 'Is it okay to'


def compute_bias(bias_func, name, actions, actions_keys):
    data_user_study_bert = dict()
    for actions_key, action in tqdm(zip(actions_keys, actions)):
        #bias_with_action, _, _ = bias_func(action)
        bias_with_action = bias_func(action)
        data_user_study_bert[actions_key] = bias_with_action[0][0]

    #csv_columns = ['Action', 'Score']

    csv_file = "./data_/correlation/pos_neg/BERT_{}_bias.csv".format(name)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            #writer.writerow(csv_columns)
            for action in actions_keys:
                data_row = [
                    data_user_study_bert[action],
                    action
                ]
                writer.writerow(data_row)
        print("Done with " + name)
    except IOError:
        print("I/O error")


emb = BERTSentence()
actions_ = actions_keys_dos.copy()
compute_bias(lambda x: emb.bias(x), 'dos_cossim', actions_, actions_)
actions_ = actions_keys_donts.copy()
compute_bias(lambda x: emb.bias(x), 'donts_cossim', actions_, actions_)
del emb

emb_sub = BERTSentenceSubspace()
norm = 8.946814
actions_ = actions_keys_dos.copy()
compute_bias(lambda x: emb_sub.bias(x, qa_template=True, norm=norm), 'dos_subspace_qa', actions_, actions_)
actions_ = actions_keys_donts.copy()
compute_bias(lambda x: emb_sub.bias(x, qa_template=True, norm=norm), 'donts_subspace_qa', actions_, actions_)

actions_ = [question + ' {}'.format(a) for a in actions_keys_dos]
compute_bias(lambda x: emb_sub.bias(x, qa_template=False, norm=norm), 'dos_subspace_raw', actions_, actions_keys_dos)
actions_ = [question + ' {}'.format(a) for a in actions_keys_donts]
compute_bias(lambda x: emb_sub.bias(x, qa_template=False, norm=norm), 'donts_subspace_raw', actions_, actions_keys_donts)
del emb_sub
