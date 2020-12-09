"""
/data/correlation/userstudy/userstudyVerbs_use_hubEmbedding.csv
/data/correlation/userstudy/BERT_cossim_bias.csv
/data/correlation/userstudy/BERT_subspace_qa_bias.csv
/data/correlation/userstudy/BERT_subspace_raw_bias.csv
/data/correlation/userstudy/userStudy_scores.csv
"""

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import csv
import os
from matplotlib import rc
import seaborn as sns

rc('font', **{'family':'sans-serif','sans-serif':['Arial']})
sns.set(style='ticks', palette='Set2')

rc('text', usetex=True)

# how to compute bias: execute file TODO

def read_bias_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    res = [[float(user_score), float(bert_score), float(mort_score), action] for (action, user_score, bert_score, mort_score) in data]
    return res


def own_plot(x, y, a=None, b=None, suffix="", text_pos=(-0.3, 1.2)):
    fontsize = 9
    x_all = x + a
    y_all = y + b

    fig = plt.figure(figsize=(4, 1.4))
    ax = plt.gca()

    # plt.axis([-80, 80, -0.8, 0.8])
    # plt.xticks([-80, -40, 0, 40, 80], size=8)
    # plt.yticks([-0.8, -0.4,0, 0.4, 0.8], size=8)
    #    plt.axvline(x=0, c='#898989', linestyle=':')
    #    plt.axhline(y=0, c='#898989', linestyle=':')
    # plt.plot(x_new,ffit(x_new))

    plt.scatter(x, y, s=5, color='#BE6F00', label='Do')
    plt.scatter(a, b, s=5, color='#00715E', label='Dont')
    plt.plot(np.unique(x_all), np.poly1d(np.polyfit(x_all, y_all, 1))(np.unique(x_all)),
             label='Correlation', color='#004E8A', gid='r = ' + str(round(pearsonr(x_all, y_all)[0], 3)))
    plt.ylim((-1.1, 1.5))
    plt.yticks(np.arange(-1, 1.1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=fontsize, direction='in')
    r = pearsonr(x_all, y_all)
    starlets = ''
    if r[1] < 0.05:
        if r[1] < 0.01:
            if r[1] < 0.001:
                starlets = '***'
            else:
                starlets = '**'
        else:
            starlets = '*'
    print(r)
    #input("Press key")
    plt.xlabel('MCM score', fontsize=fontsize-1)
    plt.ylabel('User Study value', fontsize=fontsize-1)
    #plt.tight_layout()
    #plt.text(-0.8, 0.12, 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=10)
    if "BERT_stsb_cossim" in suffix:
        plt.title("\\textbf{BERT$_{stsb}$ (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    elif "BERT_cossim" in suffix:
        plt.title("\\textbf{BERT (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    elif "BERT_subspace_qa" in suffix:
        plt.title("\\textbf{BERT (Moral Compass QT)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
        #plt.xlim((-1.1, 1.1))
    elif "BERT_subspace_raw" in suffix:
        plt.title("\\textbf{BERT (Moral Compass)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
        #plt.xlim((-1.1, 1.1))
    elif "glove" in suffix:
        plt.title("\\textbf{GloVe (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    else:
        plt.title("\\textbf{USE (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.grid(True, linestyle=':')
    os.makedirs('mort/plot_corr/userstudy/plots/', exist_ok=True)
    plt.savefig('mort/plot_corr/userstudy/plots/correlation_{}.svg'.format(suffix), bbox_inches='tight', dpi=600)
    #plt.show()
    plt.clf()
    plt.close()
    #exit()



#input("Press key")
def _corr(dos, donts, model_name, text_pos):
    # read dos based on input does
    with open("data/correlation/userstudy/{}_bias.csv".format(model_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        d = list(reader)
        try:
            tmp = float(d[-1][0])
            dim_score = 0
            dim_action = 1
        except:
            dim_score = 1
            dim_action = 0
        data_mcm = [x for x in d if x[dim_action] in list(zip(*dos))[0]]
    data_user = dos

    # sort mcm data and weat
    data_user.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[dim_action])

    assert len(data_user) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_user):
        assert d_m[dim_action] == d_w[0]

    data_dos = [[d_w[0], float(d_w[1]), float(d_m[dim_score])] for (d_m, d_w) in zip(data_mcm, data_user)]

    # read donts based on input donts
    with open("data/correlation/userstudy/{}_bias.csv".format(model_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        d = list(reader)
        data_mcm = [x for x in d if x[dim_action] in list(zip(*donts))[0]]

    data_user = donts

    # sort mcm data and weat
    data_user.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[dim_action])

    assert len(data_user) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_user):
        assert d_m[dim_action] == d_w[0]

    data_donts = [[d_m[0], float(d_w[1]), float(d_m[dim_score])] for (d_m, d_w) in zip(data_mcm, data_user)]

    x = [b[2] for b in data_dos] # user dos
    y = [[b[0], b[1]] for b in data_dos] # action + mcm does
    a = [b[2] for b in data_donts] # user donts
    b = [[b[0], b[1]] for b in data_donts] # action + mcm donts

    """for elem in y:
        print(elem)
    for elem in b:
        print(elem)"""

    y = [p[1] for p in y] # mcm does
    b = [p[1] for p in b] # mcm donts

    own_plot(x, y, a, b, suffix="userstudy_vs_{}".format(model_name), text_pos=text_pos)
    #weat_vs_BERTprojqt
    #weat_vs_BERTprojraw

    f = np.array(x + a)
    d = np.array(y + b)
    #print('###', pearsonr(x, y))
    print('Pearson ###', pearsonr(f, d))

    #print(np.mean(x))
    #print(np.mean(a))
    #print(np.mean(f))
    #print("Mean and std all", np.mean(x + a), np.std(x + a))
    #print("Mean and std do", np.mean(x), np.std(x))
    #print("Mean and std dont", np.mean(a), np.std(a))


if __name__ == '__main__':
    with open("data/correlation/userstudy/userStudy_scores.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    data = [[action, (float(user_score) - 0.5) / 0.5] for (action, user_score) in data]

    #seperate dos and donts based on user study scores
    dos = [[action, float(user_score)] for (action, user_score) in data if float(user_score) >= 0.]
    donts = [[action, float(user_score)] for (action, user_score) in data if float(user_score) < 0.]

    # glove cossim
    print("GloVe")
    text_pos = (0.11, 1.2,)
    _corr(dos, donts, 'glove_cossim', text_pos)
    # USE cossim
    print("USE")
    text_pos = (-0.035, 1.2,)
    _corr(dos, donts, 'USE', text_pos)
    # BERT stsb cossim
    #text_pos = (0.11, 1.1,)
    #_corr(dos, donts, 'BERT_stsb_cossim', text_pos)
    # BERT cossim
    print("BERT cossim")
    text_pos = (-0.13, 1.2,)
    _corr(dos, donts, 'BERT_cossim', text_pos)
    # BERT proj
    print("BERT proj")
    text_pos = (-0.25, 1.2,)
    _corr(dos, donts, 'BERT_subspace_qa', text_pos)
    # BERT proj raw
    print("BERT proj raw")
    text_pos = (-0.25, 1.2,)
    _corr(dos, donts, 'BERT_subspace_raw', text_pos)
