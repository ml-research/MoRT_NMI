import matplotlib.pyplot as plt
import pandas as pd
from mort.dataMoral import get_actions_temporal, get_actions_verbs_projection, get_actions_projection, experimental_quests_paper, get_actions_verbs_allBooksNews
from mort.funcs_mcm import BERTSentence, get_sen_embedding_from
from tqdm import tqdm
import numpy as np
import argparse
import csv
import os
import pickle
from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib import rc
rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--data_cluster', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--data', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--model', default=None, type=str,
                    help='model name: bert, use_hub, use_rcv1, ...', required=True)
parser.add_argument('--cluster', default=None, type=int,
                    help='num cluster', required=True)
parser.add_argument('--dim', default=None, type=int,
                    help='dimension of embedding', required=True)
parser.add_argument('--bert_model_name', default="bert-large-nli-mean-tokens", type=str,
                    help='data name')
# parser.add_argument('--projection', help='project emb to 2D', action='store_true')


colors_a = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#99e691',
            '#17becf', '#f0d213', '#F79F1F', '#EE5A24', '#EA2027']


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


"""dim_reduce_model = None

def dim_reduce(method="pca"):
    if method == "pca":
        dim_reduce_model = PCA(n_components=2, svd_solver='full')
"""

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_path = None
    if len(args.model.split('_')) > 1:
        model_name = args.model.replace(args.model.split('_')[0] + "_", "")
        checkpoint_path = 'retrain_use/skip_thoughts/trained_models/train_' + model_name + '/'

    transormer_model = args.bert_model_name
    transormer_model_name = os.path.basename(transormer_model) if os.path.isdir(transormer_model) else transormer_model

    filename = './mort/results/bias/{}Verbs_{}Embedding_{}.csv'.format(args.data_cluster, args.model, transormer_model_name)
    filename_pickled_cluster = filename.replace("/bias/", "/bias/cluster/")
    filename_pickled_cluster = filename_pickled_cluster.replace(".csv", "_Cluster{}_Embdim{}_PCA.p".format(args.cluster,
                                                                                                       args.dim))

    if args.model == 'use_hub':
        get_sen_embedding_ = get_sen_embedding_from('use_hub')
    else:

        emb = BERTSentence(device="cuda", transormer_model=transormer_model)
        get_sen_embedding_ = emb.get_sen_embedding

    #
    if not os.path.isfile(filename_pickled_cluster):
        # data = get_actions_verbs()
        if args.data_cluster == 'context':
            data_verbs = get_actions_projection()
        elif args.data_cluster == 'atomic':
            data_verbs = get_actions_verbs_projection()
        elif args.data_cluster == 'all':
            data_verbs = get_actions_temporal(args.data_cluster)
        else:
            raise ValueError("data not found: {}".format(args.data_cluster))

        questions = [q[0] for q in experimental_quests_paper]
        data = [questions[0].format(d) for d in data_verbs]

        res = list()
        n = 5
        for i in tqdm(range(int(len(data) // n) + 1)):
            batch = data[i * n: i * n + n]
            res += get_sen_embedding_(batch, dtype='list')

        assert len(res) == len(data)

        res = np.array(res)
        for q_idx in tqdm(range(1, len(questions))):
            data = [questions[q_idx].format(d) for d in data_verbs]
            res_tmp = list()
            for i in tqdm(range(int(len(data) // n) + 1)):
                batch = data[i * n: i * n + n]
                res_tmp += get_sen_embedding_(batch, dtype='list')
            res_tmp = np.array(res_tmp)
            res += res_tmp
            # res = [r + res_tmp[idx] for idx, r in enumerate(res)]
            assert len(res) == len(data)
        res = np.array([np.array(r) / len(questions) for r in res])

        n_components = args.dim

        dim_reduc = PCA(n_components=n_components, svd_solver='full')
        Y = dim_reduc.fit_transform(res)

        res = Y
        corpus_embeddings = (res, data_verbs)
        # Perform kmean clustering

        import scipy.cluster.hierarchy as shc

        """for idx, verb in enumerate(data_verbs):
            print(idx, verb)
        plt.figure(figsize=(10, 7))
        plt.title("Customer Dendograms")

        dend = shc.dendrogram(shc.linkage(corpus_embeddings[0], method='ward'))"""
        # plt.show()

        from sklearn.cluster import AgglomerativeClustering as Clustering

        # from sklearn.cluster import SpectralClustering as Clustering

        num_clusters = args.cluster

        clustering_model = Clustering(n_clusters=num_clusters)
        # clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.001)
        clustering_model.fit(corpus_embeddings[0])
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(data_verbs[sentence_id])

        """
        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(cluster)
            print("")
        """

        biasDictAtomic = dict()
        #os.makedirs(os.path.dirname(filename), exist_ok=True)
        """with open(filename, 'r') as f:
            reader = csv.reader(f)
            biasList_atomic = list(reader)

            for i in range(len(biasList_atomic)):
                if biasList_atomic[i][1] in biasDictAtomic.keys():
                    biasDictAtomic[biasList_atomic[i][1]].append(float(biasList_atomic[i][0]))
                else:
                    biasDictAtomic[biasList_atomic[i][1]] = list()
                    biasDictAtomic[biasList_atomic[i][1]].append(float(biasList_atomic[i][0]))"""

        action_colors_dict = dict()
        for idx, cluster in enumerate(clustered_sentences):
            #cluster_mean_bias = 0
            for action in cluster:
                action_colors_dict[action] = colors_a[idx]
                #cluster_mean_bias += biasDictAtomic[action][0]
            cluster_mean_emb = np.array(
                [e for (e, a) in zip(corpus_embeddings[0], corpus_embeddings[1]) if a in cluster])
            assert len(cluster_mean_emb) == len(cluster)

            cluster_mean_emb = np.mean(cluster_mean_emb, axis=0)

            #cluster_mean_bias /= len(cluster)
            clustered_sentences[idx] = (cluster, cluster_mean_emb)

        # clustered_sentences.sort(key=lambda x: -x[1])
        actions_colors = list()
        for a in corpus_embeddings[1]:
            actions_colors += [action_colors_dict[a]]
        moral_score_sign = clustered_sentences[1]if "kill" in clustered_sentences[0][0] else clustered_sentences[0]
        moral_score_sign = np.sign(moral_score_sign[1][0])
        moralprojection_model = dict()
        moralprojection_model["clustered_sentences"] = clustered_sentences
        moralprojection_model["projection"] = dim_reduc
        moralprojection_model["Y"] = Y
        moralprojection_model["actions_colors"] = actions_colors
        moralprojection_model["corpus_embeddings"] = corpus_embeddings
        moralprojection_model["sign"] = moral_score_sign

        os.makedirs(os.path.dirname(filename_pickled_cluster), exist_ok=True)
        pickle.dump(moralprojection_model, open(filename_pickled_cluster, "wb"))
    else:
        moralprojection_model = pickle.load(open(filename_pickled_cluster, "rb"))
        clustered_sentences = moralprojection_model["clustered_sentences"]
        dim_reduc = moralprojection_model["projection"]
        Y = moralprojection_model["Y"]
        actions_colors = moralprojection_model["actions_colors"]
        corpus_embeddings = moralprojection_model["corpus_embeddings"]

    # plt.show()

    from matplotlib.ticker import NullFormatter

    # Y = tsne.fit_transform(corpus_embeddings[0])
    # Y = res
    fig, ax = plt.subplots(figsize=(20, 12))
    texts = []
    plt.scatter(Y[:, 0], Y[:, 1], c=[colors_a[0] if t <= 0 else colors_a[1] for t in Y[:, 0]])
    #ax.xaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    plt.ylabel("2. PC", fontsize=26)
    plt.xlabel("1. PC", fontsize=26)

    plt.axis('tight')
    for i, txt in enumerate(corpus_embeddings[1]):
        texts.append(ax.text(Y[i][0], Y[i][1], txt, fontSize="26"))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axvline(x=0, linewidth=2, color='r', ls="--", alpha=0.6)
    ax.axhline(y=0, linewidth=2, color='r', ls="--", alpha=0.6, xmin=0.01, xmax=0.99)
    y_x = Y[:, 0]
    x_max = np.max(y_x)
    x_min = np.min(y_x) #- 0.6
    y_max = np.max(Y[:, 1])
    #plt.scatter([x_max + 0.7], [0], marker=">", color='r', alpha=0.6, s=80)
    #plt.scatter([x_min - 0.1], [0], marker="<", color='r', alpha=0.6, s=80)
    texts.append(ax.text(4., -1., "\\textbf{BERT's moral dimension}", fontSize="26", color='r', alpha=0.6))
    ax.annotate("\\textbf{Don'ts}", (x_max, y_max), fontSize="26")
    ax.annotate("\\textbf{Dos}", (x_min, y_max), fontSize="26")
    #ax.tick_params(axis='both', which='major', labelsize=26, direction='in')
    #ax.tick_params(axis='both', which='minor', labelsize=26, direction='in')
    adjust_text(texts, only_move={'points': 'y', 'text': 'y', 'objects': 'xy'}, lim=500)
    ensure_dir(filename_pickled_cluster.replace(".p", "/figures/moral_projection.pdf"))
    plt.savefig(filename_pickled_cluster.replace(".p", "/figures/moral_projection.svg"), bbox_inches='tight')
    plt.close()
    plt.clf()

    """clustered_sentences_print = clustered_sentences.copy()
    clustered_sentences_print.sort(key=lambda x: -x[1])
    for i, (cluster, bias, mean_emb) in enumerate(clustered_sentences_print):
        print("Cluster ", i + 1, "Mean Bias ", bias)
        print(cluster)
        print("")"""

    from scipy import spatial

    verbose = False

    if args.data == 'context':
        test_actions = get_actions_projection()
    elif args.data == 'atomic':
        test_actions = get_actions_verbs_projection()
    elif args.data == 'allNews':
        test_actions = get_actions_temporal(args.data)
    elif args.data == 'all':
        test_actions = get_actions_verbs_allBooksNews()
    else:
        raise ValueError("data not found: {}".format(args.data_cluster))

    print(dim_reduc.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', which='major', labelsize=26, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=26, direction='in')

    plt.bar(np.arange(len(dim_reduc.explained_variance_ratio_[:5])), dim_reduc.explained_variance_ratio_[:5])
    plt.savefig(filename_pickled_cluster.replace(".p", "/figures/pca_varianceratio.pdf"), bbox_inches='tight')
    plt.close()
    plt.clf()

    fig, ax = plt.subplots(figsize=(20, 12))
    texts = []
    #ax = plt.gca()
    #plt.scatter(Y[:, 0], Y[:, 1], c=actions_colors)
    #for i, txt in enumerate(corpus_embeddings[1]):
    #    texts.append(ax.text(Y[i][0], Y[i][1], txt, fontSize="22"))


    #plt.show()
    #plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
    # USE: [0.1211112  0.07854936 0.07095485 0.05513662 0.04408438 0.04262063
    #  0.03994997 0.03720206 0.03313266 0.031493  ]
    # BERT [0.25640591 0.07865071 0.06704894 0.05975881 0.05017664 0.04586655
    #  0.04264569 0.03742053 0.03427195 0.02892201]
    #1 / 0

    res = list()
    for test_action in tqdm(test_actions):
        test_questions = [q[0].format(test_action) for q in experimental_quests_paper]
        test = get_sen_embedding_(test_questions, dtype='list')
        test = np.mean(test, axis=0)
        test = dim_reduc.transform([test])[0]
        #test = octave.run_data_through_network(network, np.array([test]))[0]
        test_plot = test
        # test_plot = dim_reduc.transform([test])[0]
        # print(test_questions)

        id_nearest_cluster = (None, None)  # id, Similarity
        for i, (cluster, mean_emb) in enumerate(clustered_sentences):
            # cos_sim = 1 - spatial.distance.cosine(mean_emb, test)
            dist = spatial.distance.euclidean(mean_emb, test)
            # if id_nearest_cluster[0] is None or id_nearest_cluster[1] <= cos_sim:
            if id_nearest_cluster[0] is None or id_nearest_cluster[1] > dist:
                id_nearest_cluster = (i, dist)

        if verbose:
            print("Action ", test_action,
                  "Closest cluster ", id_nearest_cluster[0],
                  "Similarity ", id_nearest_cluster[1],
                  )
        # print(clustered_sentences[id_nearest_cluster[0]][0])

        res += [[test_action, test_plot[0], test_plot[1]]]
        y_pos = test_plot[1]#*(np.abs(test_plot[1]+1))
        plt.scatter([test_plot[0]], [y_pos], c=[colors_a[0] if test_plot[0] <= 0 else colors_a[1]], marker="x") # test_plot[1] just for viz

        texts.append(ax.text(test_plot[0], test_plot[1], test_action, fontSize="26"))

    res.sort(key=lambda x: x[1])

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axvline(x=0, linewidth=2, color='r', ls="--", alpha=0.6)
    ax.axhline(y=0, linewidth=2, color='r', ls="--", alpha=0.6, xmin=0.05, xmax=0.95)
    y_x = [float(t[1]) for t in res]
    y_y = [float(t[2]) for t in res]
    x_max = np.max(y_x)
    x_min = np.min(y_x)# - 0.6
    y_max = np.max(y_y)
    #ax.scatter([x_max + 3.], [0], marker=">", color='r', alpha=0.6, s=80)
    #ax.scatter([x_min - 4.], [0], marker="<", color='r', alpha=0.6, s=80)
    texts.append(ax.text(4., -1., "\\textbf{BERT's moral dimension}", fontSize="26", color="r", alpha=0.6))
    ax.annotate("\\textbf{Don'ts}", (x_max, y_max), fontSize="26")
    ax.annotate("\\textbf{Dos}", (x_min, y_max), fontSize="26")
    #ax.xaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    #plt.title("Querying Moral Projection PCA using {}".format("USE" if "use" in args.model else "BERT"))
    ax.tick_params(axis='both', which='major', labelsize=26, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=26, direction='in')

    plt.ylabel("2. PC", fontsize=26)
    plt.xlabel("1. PC", fontsize=26)
    plt.axis('tight')
    #adjust_text(texts, only_move={'points':'y', 'text':'y', 'objects':'xy'}, lim=700, expand_text=(1.05, 1.2),
    #            precision=0.001)
    adjust_text(texts, only_move={'points': 'y', 'text': 'y', 'objects': 'xy'}, lim=500)
    plt.savefig(filename_pickled_cluster.replace(".p", "/figures/moral_projection_query.svg"), bbox_inches='tight')
    plt.close()
    plt.clf()

    # for elem in res:
    # print(elem[1], ":", elem[0])
    #

    # save data to file
    if False:
        filename = 'mort/results/bias/cluster/{}Verbs_{}_Embedding_{}_Cluster{}_Embdim{}{}Verbs_PCA.csv'.format(args.data,
                                                                                                             transormer_model_name,
                                                                                                              args.model,
                                                                                                              args.cluster,
                                                                                                              args.dim,
                                                                                                              args.data_cluster)
        ensure_dir(filename)
        with open(filename, mode='w') as result_file:
            result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for elem in res:
                result_writer.writerow([elem[0], elem[1], round(elem[2], 4)])

        res = list(zip(corpus_embeddings[1], Y[:, 0]))
        res.sort(key=lambda x: x[1])
        with open(filename.replace(".csv", "atomicVerbs.csv"), mode='w') as result_file:
            result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for elem in res:
                result_writer.writerow([elem[0], round(elem[1], 4)])

        print('****** file has been saved.. to', filename)

    """
    # save data to file
    filename = 'experiments/results/{}Verbs_{}Embedding_{}.tsv'.format(args.data_cluster, args.model, "{}")

    X = np.array(res)
    df = pd.DataFrame(X)
    df.to_csv(filename.format("emb"), sep="\t", index=False, header=False)"""
