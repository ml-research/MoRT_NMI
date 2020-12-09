import csv
import matplotlib.pyplot as plt
import numpy as np
from mcm.funcs_mcm import normalizeBias3
from mcm.dataMoral import actions_paper as actions, frontiers_template_action_mapping
import os
from matplotlib import rc

rc('text', usetex=True)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


colors_lines = ["#009432",
                "#FFC312",
                "#EE5A24",
                "#D980FA",
                "#A3CB38",
                "#9980FA",
                "#006266",
                "#eb7394",
                '#F79F1F', '#5758BB', '#EA2027'
                # "#FDA7DF",
                # "#ED4C67",
                # "#C4E538",
                ]

colors_markers = ["#F79F1F",
                  "#12CBC4",
                  "#0652DD",
                  "#833471",
                  "#1B1464",
                  "#9980FA",
                  "#006266",
                  "#eb7394",
                  '#F79F1F', '#5754ff', '#EA2027'
                  ]


def smooth_data(data, weight=0.9):  # Weight between 0 and 1
    last = data[0]  # First value in the plot (first timestep)
    smoothed = np.zeros_like(data)
    for i, point in enumerate(data):
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed[i] = smoothed_val  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


# data = smooth_data(data, weight=0.9)

def rank_data(data, baseline):
    #print(np.array(data).shape)
    #print(np.array(baseline).shape)

    return np.array(data) - np.array(baseline)


def getlabel(name):
    a = name.split("_")
    if len(a) == 3:
        return 'Books\n{}-{}'.format(a[1], a[2])
    else:
        if a[1] == 'hub':
            return 'USE'
        elif a[1] == 'trc2':
            return 'News\n(2008-09)'
        elif a[1] == 'rcv1':
            return 'News\n(1996-97)'
        elif a[1] == '1987':
            return 'News\n(1987)'
        elif a[1] == 'rc':
            return 'Religous \&\nConstitution'
        elif a[1] == 'news':
            return 'News'
        else:
            raise ValueError("Unsupported Dataset")


def plot_bias(tempNameList, smooth=False, rank=False):
    biasDict = dict()
    biasDictAtomic = dict()

    normalizeBy = "allcorpora"
    dataName = 'context'

    print(tempNameList)
    current_tempListName = tempNameList[0]
    tempNameList = tempNameList[1]
    for current_tempName in tempNameList:
        filename = './experiments/results/bias/{}_{}Embedding.csv'.format("{}", current_tempName)

        with open(filename.format(normalizeBy + "Verbs"), 'r') as f:
            reader = csv.reader(f)
            biasListAll = [row[0] for row in list(reader)]
            maxDataAllAbs = np.max(np.abs(np.array(biasListAll[:], dtype=np.float32)))
            maxDataAll = np.max(np.array(biasListAll[:], dtype=np.float32))
            minDataAll = np.min(np.array(biasListAll[:], dtype=np.float32))

        ### context
        with open(filename.format('contextVerbs'.format(dataName)), 'r') as f:
            reader = csv.reader(f)
            biasList = list(reader)

            biasList = normalizeBias3(biasList, maxData=maxDataAll, minData=minDataAll)
            # normalize by all data
            for i in range(len(biasList)):
                if biasList[i][1] in biasDict.keys():
                    biasDict[biasList[i][1]].append(float(biasList[i][0]))
                else:
                    biasDict[biasList[i][1]] = list()
                    biasDict[biasList[i][1]].append(float(biasList[i][0]))

        ### atomic
        with open(filename.format('atomicVerbs'.format(dataName)), 'r') as f:
            reader = csv.reader(f)
            biasList_atomic = list(reader)

            biasList_atomic = normalizeBias3(biasList_atomic, maxData=maxDataAll, minData=minDataAll)

            for i in range(len(biasList_atomic)):
                if biasList_atomic[i][1] in biasDictAtomic.keys():
                    biasDictAtomic[biasList_atomic[i][1]].append(float(biasList_atomic[i][0]))
                else:
                    biasDictAtomic[biasList_atomic[i][1]] = list()
                    biasDictAtomic[biasList_atomic[i][1]].append(float(biasList_atomic[i][0]))

    line_colors = colors_lines
    xlabels = [getlabel(name) for name in tempNameList]

    for (base, contexts) in actions:
        if not base == 'have a gun':
            base_verb = base.split(" ")[0]
        else:
            base_verb = base

        if base_verb not in frontiers_template_action_mapping[current_tempListName]:
            continue

        contexts = contexts.split(",")
        fig = plt.figure(figsize=(28 if len(tempNameList) > 4 else 18, 14 if len(tempNameList) > 4 else 10))
        ax = fig.add_subplot(111)
        lines = list()

        biases = biasDictAtomic[base_verb]

        if smooth:
            biases = smooth_data(biases, weight=0.5)
        biases_base = biases.copy()
        if rank:
            biases = rank_data(biases, biases_base)
        current_line, = ax.plot(range(len(tempNameList)), biases, lw=8.0, ls="--", c='#5a5b52', label=base_verb)

        for i in range(len(tempNameList)):
            _, = ax.plot([i], biases[i:i + 1],
                         linewidth=5.0, marker="o", mew=20., mfc='#5a5b52', mec='#5a5b52')  # >

        if len(tempNameList) > 4:
            plt.xticks(range(len(tempNameList)), xlabels,
                        rotation='vertical', va='center')
        else:
            plt.xticks(range(len(tempNameList)), xlabels,
                       va='center')

        ax.tick_params(axis='both', which='major', labelsize=48)
        ax.tick_params(axis='both', which='minor', labelsize=38)

        plt.grid(True)

        for word_idx, context in enumerate(contexts):
            # We define a fake subplot that is in fact only the plot.
            verb = base + " " + context
            if verb == "go home":
                continue
            biases = biasDict[verb]
            if smooth:
                biases = smooth_data(biases, weight=0.5)
            if rank:
                biases = rank_data(biases, biases_base)

            current_line, = ax.plot(range(len(tempNameList)), biases, lw=18.0, ls="-", c=line_colors[word_idx],
                                    label=verb,
                                    alpha=0.5)

            for i in range(len(tempNameList)):
                markers_a, = ax.plot([i], biases[i:i + 1],
                                     linewidth=5.0, marker="o", mew=28., mfc=line_colors[word_idx],
                                     mec=line_colors[word_idx],
                                     alpha=1.0)

        if len(tempNameList) > 4:
            bbox_to_anchor_h = 1.25
            if len(ax.lines)> 80:
                bbox_to_anchor_h=1.32
            ax.legend(fontsize="48", loc='upper center', bbox_to_anchor=(0.5, bbox_to_anchor_h),
                      ncol=3, fancybox=True, shadow=False, framealpha=0.7)
        else:
            bbox_to_anchor_h = 1.52
            if len(ax.lines) > 20:
                bbox_to_anchor_h = 1.62
            if len(ax.lines) > 30:
                bbox_to_anchor_h = 1.72
            ax.legend(fontsize="48", loc='upper center', bbox_to_anchor=(0.5, bbox_to_anchor_h),
                      ncol=2, fancybox=True, shadow=False, framealpha=0.7)

        labels = ax.set_xticklabels(xlabels)
        for i, label in enumerate(labels):
            # pass
            label.set_y(label.get_position()[1] - 0.133)

        file_save_path = "./experiments/results/temp_results_frontiers/{}/{}/{}Merged{}{}/plot_{}.svg".format(normalizeBy,
                                                                                                    "".join(current_tempListName),
                                                                                                    dataName,
                                                                                                    "Smoothed" if smooth else "",
                                                                                                    "Ranked" if rank else "",
                                                                                                    base.replace(" ",
                                                                                                                 "_"))

        ensure_dir(file_save_path)
        plt.savefig(file_save_path, bbox_inches="tight")
        plt.clf()
        plt.close()


if __name__ == '__main__':
    # books decade
    tempNameList = ['decade', ['use_1800_1809', 'use_1810_1819', 'use_1820_1829', 'use_1830_1839', 'use_1840_1849',
                    'use_1850_1859', 'use_1860_1869', 'use_1870_1879', 'use_1880_1889', 'use_1890_1899']]
    plot_bias(tempNameList, smooth=True)
    # news
    tempNameList = ['news', ['use_1987', 'use_rcv1', 'use_trc2']]
    plot_bias(tempNameList, rank=True, smooth=False)
    plot_bias(tempNameList, rank=False, smooth=False)
    # books century
    tempNameList = ['century', ['use_1510_1699', 'use_1700_1799', 'use_1800_1899']]
    plot_bias(tempNameList, rank=True, smooth=False)
    plot_bias(tempNameList, rank=False, smooth=False)
    # news vs books vs rc
    # trc2 vs books vs rc
    tempNameList = ['diff2', ['use_rc', 'use_1800_1899', 'use_trc2']]
    plot_bias(tempNameList, rank=True)
    plot_bias(tempNameList, rank=False)

