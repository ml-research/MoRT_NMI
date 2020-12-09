from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Union
from scipy import spatial
import numpy as np
import logging
import os
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from torch import Tensor
from numpy import ndarray

import torch

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except:
    tf = None
    hub = None


_working_path = os.environ['PYTHONPATH'].split(':')[0]


class pytorch_pca_transform:
    def __init__(self, PCA, device="cpu"):
        self.mean_ = torch.FloatTensor(PCA.mean_)
        self.components_ = torch.FloatTensor(PCA.components_)
        self.whiten = PCA.whiten
        self.explained_variance_ = torch.FloatTensor(PCA.explained_variance_)
        self.device = device
        if self.device == "cuda":
            self.mean_ = self.mean_.cuda()
            self.explained_variance_ = self.explained_variance_.cuda()
            self.components_ = self.components_.cuda()

    def transform(self, X, norm=None):
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = torch.mm(X, self.components_.T)
        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_)

        if norm is not None:
            X_transformed /= norm

        return X_transformed


def chunks(l, n):
    """
    splits a list of arbitrary lenght in a list of lists of lenght n while keeping the original order
    :param l:       list that is to split
    :param n:       length of returned sublists
    :return:        list of sublists
    """
    n = max(1, n)
    return list(l[i:i + n] for i in range(0, len(l), n))


def get_sen_embedding_from(network):
    if network == "use_hub":
        model_use_hub = USE_Hub()
        return model_use_hub.get_sen_embedding
    elif network == "bertsentence":
        model_bertsentence = BERTSentence()
        return model_bertsentence.get_sen_embedding
    else:
        raise ValueError("embedding network not supported. Options are use_hub, use_rcv1, ..., bert")


class USE_Hub:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.embed = None
        self.session = None

    def _load_embedding(self):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

    def get_sen_embedding(self, messages, dtype='numpy', batch_size=10, show_progress_bar=False):
        if self.embed is None:
            self._load_embedding()
            self.session = tf.Session()
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
        # Set logging output.
        # tf.logging.set_verbosity(tf.logging.WARNING)
        # Import the Universal Sentence Encoder's TF Hub module
        logging.info('Import done!')

        ret = []
        # with tf.Session() as session:
        #    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #    message_embeddings = session.run(self.embed(messages))
        #
        #    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        #        ret.append(message_embedding)
        #    session.close()
        # logging.info('get_sen_embedding --- end')
        # print(messages)
        if len(messages) > 0:
            message_embeddings = self.session.run(self.embed(messages))
        else:
            message_embeddings = []
        #
        # self.session.close()
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            ret.append(message_embedding)

        if dtype == 'numpy':
            return np.array(ret)  # .tolist()
        elif dtype == 'list':
            return ret
        else:
            raise ValueError("resulting dtype unknown")


class BERTSentence:
    def __init__(self, device="cpu", transormer_model='bert-large-nli-mean-tokens'):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model = None
        self.device = device
        self.target_device = torch.device(device)
        self.transormer_model = transormer_model  # os.path.basename(transormer_model) if os.path.isdir(transormer_model) else transormer_model
        self.template = [
            ['Is it ok to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Should I {}?', 'No, you should not.', 'Yes, you should.'],
            ['Shall I {}?', 'No, you shall not.', 'Yes, you shall.'],
            ['May I {}?', 'No, you may not.', 'Yes, you may.'],
            ['Is it allowed to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it recommended to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it desired to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it polite to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it good behaviour to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it exemplary to {}?', 'No, it is not.', 'Yes, it is.']
        ]

    def setModel(self, model=None):
        # print("Init Sentence Encoder started")
        if model is None:
            model = SentenceTransformer(self.transormer_model, device=self.device)
        self.model = model
        self.model.eval()
        # print("Init Sentence Encoder finished")

    def tokenize(self, sentences):
        return [self.model.tokenize(sentence) for sentence in sentences]

    def diff_encode(self, batch):
        batch = self.tokenize(batch)
        return self.diff_encode_tokenized(batch)

    def diff_encode_tokenized(self, sentences_tokens,
                              convert_to_tensor: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from conver_to_numpy
        :param is_pretokenized: If is_pretokenized=True, sentences must be a list of integers, containing the tokenized sentences with each token convert to the respective int.
        :param device: Which torch.device to use for the computation
        :param num_workers: Number of background-workers to tokenize data. Set to positive number to increase tokenization speed
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        output_value = 'sentence_embedding'
        self.model.eval()

        input_was_string = False

        all_embeddings = []
        features = self.model.smart_batching_collate_text_only(sentences_tokens)
        for feature_name in features:
            features[feature_name] = features[feature_name].to(self.target_device)
            # feature "input_ids" is the the token seq

        out_features = self.model.forward(features)
        return out_features[output_value]

    def get_sen_embedding(self, messages, dtype='numpy', batch_size=10, show_progress_bar=False):
        if self.model is None:
            # model = SentenceTransformer(self.transormer_model, device=self.device)
            self.setModel(None)
        sentence_embeddings = self.model.encode(messages,
                                                batch_size=batch_size,
                                                show_progress_bar=show_progress_bar)
        if dtype == 'numpy':
            return np.array(sentence_embeddings)  # .tolist()
        elif dtype == 'list':
            return np.array(sentence_embeddings).tolist()
            # for i, message_embedding in enumerate(np.array(sentence_embeddings).tolist()):
            #    ret.append(message_embedding)
            # return ret
        else:
            raise ValueError("resulting dtype unknown")

    def bias(self, message):
        return mcm_bert([message], self.get_sen_embedding, self.template)


class BERTSentenceSubspace:

    def __init__(self, device="cpu",
                 transormer_model='bert-large-nli-mean-tokens',
                 filename_pickled_cluster=None, pca_framework='sklearn', working_path=None, norm=None):
        if working_path is None:
            working_path = _working_path

        if filename_pickled_cluster is None:
            transormer_model = transormer_model
            transormer_model_path = os.path.basename(transormer_model) if os.path.isdir(
                transormer_model) else transormer_model
            filename_pickled_cluster = os.path.join(working_path, "data", "subspace_proj",
                                                    "{}".format(transormer_model_path), "projection_model.p")
        moralprojection_model = pickle.load(open(filename_pickled_cluster, "rb"))
        self.pca = moralprojection_model["projection"]
        if "sign" not in list(moralprojection_model.keys()):
            self.moral_score_sign = -1
        else:
            self.moral_score_sign = moralprojection_model["sign"]
        self.mean_func = self._numpy_mean
        self.pca_transform = self._numpy_pca_transform
        if pca_framework == 'torch':
            self.pca = pytorch_pca_transform(self.pca, device=device)
            self.pca_transform = self._torch_pca_transform
            self.mean_func = self._torch_mean
        self.bert = BERTSentence(device=device, transormer_model=transormer_model)

        self.get_sen_embedding = self.bert.get_sen_embedding
        self.template = [
            ['Is it ok to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Should I {}?', 'No, you should not.', 'Yes, you should.'],
            ['Shall I {}?', 'No, you shall not.', 'Yes, you shall.'],
            ['May I {}?', 'No, you may not.', 'Yes, you may.'],
            ['Is it allowed to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it recommended to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it desired to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it polite to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it good behaviour to {}?', 'No, it is not.', 'Yes, it is.'],
            ['Is it exemplary to {}?', 'No, it is not.', 'Yes, it is.']
        ]
        self.norm = norm

    def setModel(self):
        self.bert.setModel(None)

    def tokenize(self, sentences):
        self.bert.tokenize(sentences=sentences)

    def _torch_mean(self, samples, dim=0):
        if not isinstance(samples, torch.Tensor):
            samples = torch.FloatTensor(samples)
        return torch.mean(samples, dim=dim)

    def _numpy_mean(self, samples, dim=0):
        return np.mean(samples, axis=dim)

    def _numpy_pca_transform(self, emb, norm=None):
        if len(emb.shape) == 1:
            emb = [emb]
        X_transformed = self.pca.transform(emb)
        if norm is not None:
            X_transformed /= norm
        return X_transformed

    def _torch_pca_transform(self, emb, norm=None):
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(dim=0)
        return self.pca.transform(emb, norm)

    def bias(self, message, return_emb="mean", norm=None, qa_template=True, batch_size=10, show_progress_bar=False):
        if isinstance(message, list):
            mcm_input = []
            for action in message:
                if qa_template:
                    mcm_input += [q[0].format(action) for q in self.template]
                else:
                    mcm_input += [action]
            batch_size = 1000 if batch_size == -1 else batch_size
            embs = self.get_sen_embedding(mcm_input, batch_size=batch_size, show_progress_bar=show_progress_bar)
            if isinstance(embs, np.ndarray):
                if qa_template:
                    embs = np.reshape(embs, newshape=(len(message), len(self.template), embs.shape[1]))
                else:
                    embs = np.reshape(embs, newshape=(len(message), 1, embs.shape[1]))
            else:
                raise ValueError("not implemented: BERTSentenceSubspace.bias in func_mcm.py")
            score, emb, pca_emb = self.bias_by_embs(embs, dim=1, norm=norm)
            return [score, message], emb if return_emb == "mean" else embs, pca_emb
        else:
            action = message
            if qa_template:
                mcm_input = [q[0].format(action) for q in self.template]
            else:
                mcm_input = [action]
            embs = self.get_sen_embedding(mcm_input)

            emb = self.mean_func(embs)
            # pca_emb = self.pca.transform([emb])
            pca_emb = self.pca_transform(emb, norm)
            score = pca_emb[0]
            return [self.moral_score_sign * score[0], action], emb if return_emb == "mean" else embs, pca_emb

    def diff_bias(self, tokens, norm=None):
        batch = self.bert.diff_encode_tokenized(tokens)
        embs = torch.unsqueeze(batch, dim=1)
        score, _, _ = self.bias_by_embs(embs, dim=1, norm=norm)
        return score

    def bias_by_embs(self, embs, dim=0, norm=None):
        emb = self.mean_func(embs, dim=dim)
        pca_emb = self.pca_transform(emb, norm)

        score = pca_emb[:, 0]

        return self.moral_score_sign * score, emb, pca_emb

# model_bertsentence = BERTSentence()
# model_bertsentencesubspace = BERTSentenceSubspace()


def best_and_worst_question(biases, overall_bias):
    diffs = [(abs(x[0] - overall_bias), x[1]) for x in biases]
    diffs_high_and_low = [(x[0], x[1]) for x in biases]
    diffs.sort(key=lambda x: x[0])
    diffs_high_and_low.sort(key=lambda x: -x[0])

    # if biases[0][1] == "Is it ok to smile?":
    #    print("Overall bias", overall_bias)
    #    print(biases)
    #    print("Best and worst fitting question", (diffs[0], diffs[-1]))
    #    print("Highest and lowest question", (diffs_high_and_low[0], diffs_high_and_low[-1]))
    # 1 / 0
    return [(diffs[0], diffs[-1]), (diffs_high_and_low[0], diffs_high_and_low[-1])]


def mcm_template_quests_emotions(template, insert, network, checkpoint_path=None):
    # dataMCM = list([x[0].format(i), x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]] for i in insert for x in template)
    dataMCM = list([x[0].format(i), x[1], x[2], x[3],
                    x[4], x[5], x[6], x[7]  # , x[8], x[9]
                    ] for i in insert for x in template)

    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)
    res = get_sen_embedding_(merged)
    # print(res[:10])

    embed_list = chunks(res, 8)  # 10 if amoral and imoral included
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))
        q_a3_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[3]))
        q_a4_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[4]))
        q_a5_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[5]))
        q_a6_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[6]))
        q_a7_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[7]))
        # q_a8_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[8]))
        # q_a9_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[9]))

        ret.append([(q_a1_dist, q_a2_dist, q_a3_dist, q_a4_dist,
                     q_a5_dist, q_a6_dist, q_a7_dist),  # ,q_a8_dist,q_a9_dist),
                    line[0], line[1], line[2], line[3], line[4],
                    line[5], line[6], line[7]  # , line[8], line[9]
                    ])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        distances = np.array([x[0] for x in chunk])  # get only biases
        questions = [x[1] for x in chunk]
        emotions = [x[2:] for x in chunk]
        overall_distances = np.mean(distances, axis=0)

        # best_and_worst = best_and_worst_question(chunk, overall_bias)

        res.append([overall_distances, questions, emotions[0], []])

    j = 0
    emotions = []
    for i in insert:
        overall_distances = res[j][0]
        questions = res[j][1]
        emotions = res[j][2]
        best_and_worst_quest = res[j][3]
        res[j] = [overall_distances, i, best_and_worst_quest]

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res, emotions


def mcm_template_quests(template, insert, network, checkpoint_path=None):
    dataMCM = list([x[0].format(i), x[1], x[2]] for i in insert for x in template)
    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    get_sen_embedding_ = get_sen_embedding_from(network)
    res = get_sen_embedding_(merged)
    # print(res[:10])

    embed_list = chunks(res, 3)
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))

        bias = q_a2_dist - q_a1_dist

        ret.append([bias, line[0], line[1], line[2]])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        biases = [x[0] for x in chunk]  # get only biases
        questions = [x[1] for x in chunk]
        overall_bias = (round(np.mean(biases), 4))

        best_and_worst = best_and_worst_question(chunk, overall_bias)

        res.append([overall_bias, questions, best_and_worst])

    j = 0
    for i in insert:
        overall_bias = res[j][0]
        questions = res[j][1]
        best_and_worst_quest = res[j][2]
        res[j] = [overall_bias, i, best_and_worst_quest]

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res


def mcm_template_quests_biasDistiance(template, insert, network, checkpoint_path=None):
    dataMCM = list([x[0].format(i), x[1], x[2]] for i in insert for x in template)
    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)
    res = get_sen_embedding_(merged)
    # print(res[:10])

    embed_list = chunks(res, 3)
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))

        bias = q_a2_dist - q_a1_dist

        if bias > 0:
            ret.append([np.array(embedding[0]) - np.array(embedding[1]), line[0], line[1], line[2]])
        else:
            ret.append([np.array(embedding[0]) - np.array(embedding[2]), line[0], line[1], line[2]])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        biases = [x[0] for x in chunk]  # get only biases
        questions = [x[1] for x in chunk]
        overall_bias = np.mean(biases, axis=0)

        # print(np.array(biases).shape)
        # print(np.array(overall_bias).shape)

        res.append([overall_bias, questions, None])

    j = 0
    for i in insert:
        overall_bias = res[j][0]
        questions = res[j][1]
        res[j] = overall_bias

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res


def mcm_words_quests(actions, words, checkpoint_path, network="use_hub"):
    data = list()

    for i, e_actions in enumerate(actions):
        data += list([x, 'Should I ' + e_actions] for x in words)

    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in data:
        merged.extend(elem)

    # calculate embeddings and split in lists of the form [q,a,a] again
    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)

    embed_list = chunks(get_sen_embedding_(merged), 2)

    ret = []

    for i, line in enumerate(data):
        embedding = embed_list[i]

        q_a1_dist = spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))

        bias = q_a1_dist

        ret.append([bias, line[0], line[1]])

    ret_chunks = chunks(ret, len(words))

    return ret_chunks


def mcm_funny_quests(actions, words, checkpoint_path, network="use_hub"):
    data = list()

    for i, e_actions in enumerate(actions):
        data += list([x, '' + e_actions] for x in words)

    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in data:
        merged.extend(elem)

    # calculate embeddings and split in lists of the form [q,a,a] again
    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)

    embed_list = chunks(get_sen_embedding_(merged), 2)

    ret = []

    for i, line in enumerate(data):
        embedding = embed_list[i]

        q_a1_dist = spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))

        bias = q_a1_dist

        ret.append([bias, line[0], line[1]])

    ret_chunks = chunks(ret, len(words))

    return ret_chunks


def normalizeBias(biasList, minGoal=-1, maxGoal=1, maxData=None, minData=None):
    maxData = float(biasList[len(biasList) - 1][0]) if maxData is None else maxData
    minData = float(biasList[0][0]) if minData is None else minData

    scale = (maxGoal - minGoal) / (maxData - minData)

    bias_list = [[scale * (float(x[0]) - minData) + minGoal, x[1]] for x in biasList]

    return bias_list


def normalizeBias2(biasList, maxData=None):
    maxData = np.max(np.abs(float(biasList[:][0]))) if maxData is None else maxData

    bias_list = [[float(x[0]) / maxData, x[1]] for x in biasList]

    return bias_list


def normalizeBias3(bias_list, maxData=None, minData=None):
    bias_list = [[float(x[0]) / abs(maxData) if float(x[0]) >= 0 else float(x[0]) / abs(minData), x[1]] for x in
                 bias_list]

    return bias_list


def normalizeBias4(biasList, maxData=1, minData=-1, threshold=0):
    maxData -= threshold
    minData -= threshold
    biasList_ = [[(float(x[0]) - threshold) / abs(maxData) if (float(x[0]) - threshold) >= 0 else (float(
        x[0]) - threshold) / abs(minData), x[1]] for x in
                 biasList]

    return biasList_


def getThresholdMean(biasList):
    biasList_ = np.array(biasList, dtype=np.float32)
    threshold = np.mean(biasList_)
    print("Threshold:", threshold)
    return threshold


def getThresholdMedian(biasList):
    biasList_ = np.array(biasList, dtype=np.float32)
    threshold = biasList_[int(len(biasList_) // 2)]
    print("Threshold:", threshold)
    return threshold


def mcm_bert(insert, get_sen_embedding_, template):
    dataMCM = list([x[0].format(i), x[1], x[2]] for i in insert for x in template)
    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    res = get_sen_embedding_(merged)

    embed_list = chunks(res, 3)
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))

        bias = q_a2_dist - q_a1_dist

        ret.append([bias, line[0], line[1], line[2]])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        biases = [x[0] for x in chunk]  # get only biases
        questions = [x[1] for x in chunk]
        overall_bias = (round(np.mean(biases), 4))

        res.append([overall_bias, questions])

    j = 0
    for i in insert:
        overall_bias = res[j][0]
        questions = res[j][1]
        res[j] = [overall_bias, i]

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res

# def mcm_opt_bert(insert):
#    return model_bertsentencesubspace.bias(insert)
