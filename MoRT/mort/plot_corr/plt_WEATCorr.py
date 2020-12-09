import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import csv
import os
from matplotlib import rc
from mort import dataMoral
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
    plt.ylim((-0.2, 0.22))
    plt.yticks(np.arange(-0.2, 0.21, 0.1))
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
    plt.ylabel('WEAT value', fontsize=fontsize-1)
    #plt.tight_layout()
    #plt.text(-0.8, 0.12, 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=10)
    print(suffix)
    if "BERTcossim" in suffix:
        plt.title("\\textbf{BERT (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    elif "BERTstsbcossim" in suffix:
        plt.title("\\textbf{BERT$_{stsb}$ (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    elif "BERTsubspace_qa" in suffix:
        plt.title("\\textbf{BERT (Moral Compass QT)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
    elif "BERTsubspace_raw" in suffix:
        plt.title("\\textbf{BERT (Moral Compass)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
    elif "glove" in suffix.lower():
        plt.title("\\textbf{GloVe (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    else:
        plt.title("\\textbf{USE (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + starlets, color='#004E8A', fontsize=fontsize-1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.grid(True, linestyle=':')
    os.makedirs('mort/plot_corr/weat_corr/plots/', exist_ok=True)
    plt.savefig('mort/plot_corr/weat_corr/plots/correlation_{}.svg'.format(suffix), bbox_inches='tight', dpi=600)
    #plt.show()
    plt.clf()
    plt.close()
    #exit()



#input("Press key")




def use_corr():
    # dos with respect to literature
    with open("data/correlation/pos_neg/pos50Verbs_use_hubEmbedding.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_mcm = list(reader)

    data_weat = dataMoral.dos_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    foo = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    # donts with respect to literature
    with open("data/correlation/pos_neg//neg50Verbs_use_hubEmbedding.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_mcm = list(reader)
    data_weat = dataMoral.donts_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    bar = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    x = [b[2] for b in foo]
    y = [[b[0], b[1]] for b in foo]
    a = [b[2] for b in bar]
    b = [[b[0], b[1]] for b in bar]

    """for elem in y:
        print(elem)
    for elem in b:
        print(elem)"""

    y = [p[1] for p in y]
    b = [p[1] for p in b]

    own_plot(x, y, a, b, suffix="weat_vs_USE", text_pos=(-0.025, 0.15,))

    f = np.array(x + a)
    d = np.array(y + b)
    print('Pearson ###', pearsonr(f, d))

    #print(np.mean(x))
    #print(np.mean(a))
    #print(np.mean(f))
    #print("Mean and std all", np.mean(x + a), np.std(x + a))
    #print("Mean and std do", np.mean(x), np.std(x))
    #print("Mean and std dont", np.mean(a), np.std(a))


def bertcossim_corr(model_name, model_name2=''):
    # dos with respect to literature
    with open("data/correlation/pos_neg/BERT{}_dos_{}_bias.csv".format(model_name2, model_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        #['-0.045', 'toast'],
        data_mcm = list(reader)
    data_weat = dataMoral.dos_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    foo = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    # donts with respect to literature
    with open("data/correlation/pos_neg/BERT{}_donts_{}_bias.csv".format(model_name2, model_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_mcm = list(reader)
    data_weat = dataMoral.donts_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    bar = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    x = [b[2] for b in foo]
    y = [[b[0], b[1]] for b in foo]
    a = [b[2] for b in bar]
    b = [[b[0], b[1]] for b in bar]

    """for elem in y:
        print(elem)
    for elem in b:
        print(elem)"""

    y = [p[1] for p in y]
    b = [p[1] for p in b]

    own_plot(x, y, a, b, suffix="weat_vs_BERT{}{}".format(model_name2.replace('_', ''),
                                                          model_name),
             text_pos=(-0.09, .15,))
    #weat_vs_BERTprojqt
    #weat_vs_BERTprojraw

    f = np.array(x + a)
    d = np.array(y + b)
    print('Pearson ###', pearsonr(f, d))

    #print(np.mean(x))
    #print(np.mean(a))
    #print(np.mean(f))
    #print("Mean and std all", np.mean(x + a), np.std(x + a))
    #print("Mean and std do", np.mean(x), np.std(x))
    #print("Mean and std dont", np.mean(a), np.std(a))


def glove_cossim_corr():
    # dos with respect to literature
    with open("data/correlation/pos_neg/glove_dos_cossim_bias.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        #['-0.045', 'toast'],
        data_mcm = list(reader)
    data_weat = dataMoral.dos_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    foo = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    # donts with respect to literature
    with open("data/correlation/pos_neg/glove_donts_cossim_bias.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_mcm = list(reader)
    data_weat = dataMoral.donts_50

    # sort mcm data and weat
    data_weat.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[1])

    assert len(data_weat) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_weat):
        assert d_m[1] == d_w[0]

    bar = [[d_m[1], float(d_w[2]), float(d_m[0])] for (d_m, d_w) in zip(data_mcm, data_weat)]

    x = [b[2] for b in foo]
    y = [[b[0], b[1]] for b in foo]
    a = [b[2] for b in bar]
    b = [[b[0], b[1]] for b in bar]

    """for elem in y:
        print(elem)
    for elem in b:
        print(elem)"""

    y = [p[1] for p in y]
    b = [p[1] for p in b]

    own_plot(x, y, a, b, suffix="weat_vs_GLOVE", text_pos=(0.09, .15,))
    #weat_vs_BERTprojqt
    #weat_vs_BERTprojraw

    f = np.array(x + a)
    d = np.array(y + b)
    print('Pearson ###', pearsonr(f, d))

    #print(np.mean(x))
    #print(np.mean(a))
    #print(np.mean(f))
    #print("Mean and std all", np.mean(x + a), np.std(x + a))
    #print("Mean and std do", np.mean(x), np.std(x))
    #print("Mean and std dont", np.mean(a), np.std(a))

if __name__ == '__main__':
    print("GloVe")
    glove_cossim_corr()
    # USE
    print("USE")
    use_corr()
    # BERT cossim
    print("BERT cossim")
    bertcossim_corr('cossim')
    # BERT stsb cossim
    #print("BERT stsb cossim")
    #bertcossim_corr('cossim', model_name2='_stsb')
    # BERT proj
    print("BERT proj")
    bertcossim_corr('subspace_qa')
    # BERT proj raw
    print("BERT proj raw")
    bertcossim_corr('subspace_raw')