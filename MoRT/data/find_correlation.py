from numpy.random import randn
import numpy as np
from scipy.stats import spearmanr, pearsonr

fname = "./data/parsed_yes_no_BERTBias.csv"
sentences_ = list()

actions = list()


with open(fname, "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        line = line.rstrip('\n')
        print(line)
        vect = line.split(',')
        acc = float(vect[1]) / (float(vect[1]) + float(vect[2]))
        if i == 1:
            bias_bert = np.array(float(vect[3]))
            bias_behavioral = np.array(acc)
        else:
            bias_bert = np.append(bias_bert, float(vect[3]))
            bias_behavioral = np.append(bias_behavioral, acc)

#print(bias_bert)
#print(bias_behavioral)


corr, _ = pearsonr(bias_bert, bias_behavioral)
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(bias_bert, bias_behavioral)
print('Spearmans correlation: %.3f' % corr)

