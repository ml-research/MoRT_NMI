from mort.funcs_mcm import BERTSentence, BERTSentenceSubspace
from tqdm import tqdm
import csv
import os

with open("./data/user_study/userStudy_yes_no.csv", "r") as f:
    actions_keys_ = [row.split(',')[0] for row in f]
    actions_keys_ = actions_keys_[1:]

questions = dict()
with open("./data/user_study/userStudy_action_template.csv", "r") as f:
    for i, row in enumerate(f):
        if i != 0:
            q, a = row.split(',')[1], row.split(',')[0]
            questions[a] = q.replace('\n', '')


def compute_bias(bias_func, name, actions, actions_keys):
    data_user_study_bert = dict()
    for actions_key, action in tqdm(zip(actions_keys, actions)):
        #bias_with_action, _, _ = bias_func(action)
        bias_with_action = bias_func(action)
        data_user_study_bert[actions_key] = bias_with_action[0][0]

    csv_columns = ['Action', 'Score']

    csv_file = "./data_/correlation/userstudy/{}_bias.csv".format(name)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for action in actions_keys:
                data_row = [
                    action,
                    data_user_study_bert[action]
                ]
                writer.writerow(data_row)
        print("Done with " + name)
    except IOError:
        print("I/O error")


emb = BERTSentence(transormer_model='average_word_embeddings_glove.840B.300d')
actions_ = actions_keys_.copy()
compute_bias(lambda x: emb.bias(x), 'glove_cossim', actions_, actions_)
del emb


emb = BERTSentence()
actions_ = actions_keys_.copy()
compute_bias(lambda x: emb.bias(x), 'BERT_cossim', actions_, actions_)
del emb

emb_sub = BERTSentenceSubspace()
norm = 8.946814
actions_ = actions_keys_.copy()
#compute_bias(lambda x: emb_sub.bias(x, qa_template=True, norm=norm), 'BERT_subspace_qa', actions_, actions_)

actions_ = [questions[a] + ' {}'.format(a) for a in actions_keys_]
compute_bias(lambda x: emb_sub.bias(x, qa_template=False, norm=norm), 'BERT_subspace_raw', actions_, actions_keys_)
del emb_sub
