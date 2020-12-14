# dirty hack to add mort to path
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../"))
from mort.funcs_mcm import BERTSentenceSubspace

mcm_model = BERTSentenceSubspace(device="cuda",
                                 transormer_model='bert-large-nli-mean-tokens',
                                 filename_pickled_cluster=None,
                                 pca_framework='sklearn')


def mcm_(query):
    return mcm_model.bias(query, norm=8.946814, qa_template=False)


with open("input.txt") as file:
    for line in file:
        print(mcm_(line.strip())[0])
