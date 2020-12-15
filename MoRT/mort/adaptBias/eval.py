import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
from torch import cuda
from gensim.models import KeyedVectors
import mort.adaptBias.model as model
import os
import pickle
import scipy.stats
import gensim
from mort.funcs_mcm import BERTSentence, BERTSentenceSubspace
import mort.dataMoral as dataMoral
from mort.adaptBias.hyperparams.hyperparams_bert import Hyperparams_rec as hp
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

torch.manual_seed(0)
#cuda.manual_seed_all(0)
np.random.seed(0)


def eval_bias_analogy(w2v):
    print('SemBias')
    bias_analogy_f = open('./SemBias/SemBias')
    definition_num = 0
    none_num = 0
    stereotype_num = 0
    total_num = 0

    sub_definition_num = 0
    sub_none_num = 0
    sub_stereotype_num = 0
    sub_size = 40
    sub_start = -(sub_size - sum(1 for line in open('./SemBias/SemBias')))

    gender_v = w2v['he'] - w2v['she']
    for sub_idx, l in enumerate(bias_analogy_f):
        l = l.strip().split()
        max_score = -100
        for i, word_pair in enumerate(l):
            word_pair = word_pair.split(':')
            pre_v = w2v[word_pair[0]] - w2v[word_pair[1]]
            score = dot(gender_v, pre_v) / (norm(gender_v) * norm(pre_v))
            if score > max_score:
                max_idx = i
                max_score = score
        if max_idx == 0:
            definition_num += 1
            if sub_idx >= sub_start:
                sub_definition_num += 1
        elif max_idx == 1 or max_idx == 2:
            none_num += 1
            if sub_idx >= sub_start:
                sub_none_num += 1
        elif max_idx == 3:
            stereotype_num += 1
            if sub_idx >= sub_start:
                sub_stereotype_num += 1
        total_num += 1
    print('definition: {}'.format(definition_num / total_num))
    print('stereotype: {}'.format(stereotype_num / total_num))
    print('none: {}'.format(none_num / total_num))

    if sub_definition_num == 0:
        print('sub definition: 0')
    else:
        print('sub definition: {}'.format(sub_definition_num / sub_size))
    if sub_stereotype_num == 0:
        print('sub stereotype: 0')
    else:
        print('sub stereotype: {}'.format(sub_stereotype_num / sub_size))
    if sub_none_num == 0:
        print('sub none: 0')
    else:
        print('sub none: {}'.format(sub_none_num / sub_size))


def de_biassing_emb(generator):
    generator.eval()
    debias_emb_txt = 'debiased_{}/gender_debiased.txt'.format(sys.argv[1])
    debias_emb_bin = 'debiased_{}/gender_debiased.bin'.format(sys.argv[1])
    w2v = \
        KeyedVectors.load_word2vec_format(hp.word_embedding,
                                          binary=hp.emb_binary)

    emb = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=300)
    print('Start generating')
    inputs = torch.split(torch.stack([torch.FloatTensor(w2v[word]) for word in w2v.vocab.keys()]), 1024)
    debias_embs = []
    for input in inputs:
        if hp.gpu >= 0:
            input = input.cuda()
        with torch.no_grad():
            debias_embs += [generator(input).data.cpu().numpy()]
    debias_embs = np.concatenate(debias_embs)
    emb.add([word for word in w2v.vocab.keys()], debias_embs)

    return emb


def eval_autoencoder(encoder, decoder, sent2embs):

    if False:
        decoder_criterion = torch.nn.MSELoss()
        emb_list = list(sent2embs.keys())
        eval_inputs = torch.FloatTensor([sent2embs[item] for item in emb_list])
        dataset = torch.utils.data.TensorDataset(eval_inputs)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            if hp.gpu >= 0:
                x = x.cuda()
            hidden = encoder(x)
            pre = decoder(hidden)
            loss = decoder_criterion(pre, x)
            total_loss += loss.item()

        total_loss /= len(dataloader)
        print("--" * 42)
        print("Loss of AE:", total_loss)
        print("--" * 42)

    emb_bert = BERTSentence(device=hp.device)
    moral_model = BERTSentenceSubspace()
    moral_model_torch = BERTSentenceSubspace(pca_framework='torch')

    # TODO reconstruction loss on more wikipedia data
    if False:
        wikipedia_sentences_path = './data/adaptBias/tmp/sent2emb_wikipedia_{}_{}.p'.format(50000, emb_bert.transormer_model)
        all_sentences = dataMoral.adaptbias_get_random_wikipedia_sentences()[:50000]
        if os.path.isfile(wikipedia_sentences_path):
            print("checking if sentence embedding is complete")
            sent2emb = pickle.load(open(wikipedia_sentences_path, "rb"))
            for sentence in tqdm(all_sentences):
                assert sentence in list(sent2emb.keys())
        else:
            print("generate sentence embedding")
            sent2emb = dict()
            batch_size_ = 64
            #all_sentences = all_sentences[:291]
            all_sentences_emb = emb_bert.get_sen_embedding(all_sentences,
                                                      batch_size=batch_size_,
                                                      show_progress_bar=True).squeeze()
            for sent_emb, sentence in zip(all_sentences_emb, all_sentences):
                sent2emb[sentence] = sent_emb

            for sentence in tqdm(all_sentences):
                assert sentence in list(sent2emb.keys())

            pickle.dump(sent2emb, open(wikipedia_sentences_path, "wb"))

        decoder_criterion = nn.MSELoss()
        err = 0
        cnt = 0
        for sentence in tqdm(list(sent2emb.keys())):
            sent_emb = sent2emb[sentence]
            sent_emb = torch.FloatTensor([sent_emb])
            sent_emb = sent_emb.cuda()
            hidden = encoder(sent_emb)
            sent_emb_reconstructed = decoder(hidden)
            err += decoder_criterion(sent_emb_reconstructed, sent_emb).data.cpu().numpy().item()
            cnt += 1
        print("Wikipedia 50000 MSRec.Err:", err/cnt)
        input("any key to continue")

    # start with eval neg and pos actions
    sentences_calc_moral_pos, sentences_calc_moral_neg = dataMoral.adaptbias_get_pos_and_neg()
    sentences_calc_moral = list(sentences_calc_moral_pos.keys())
    print("--" * 42)
    print("Atomic actions pos", len(sentences_calc_moral))
    print("--" * 42)
    for i, message in enumerate(sentences_calc_moral):
        break
        bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
        embs = torch.FloatTensor(embs)
        embs = embs.cuda()
        hidden = encoder(embs)
        embs_reconstructed = decoder(hidden)
        bias_with_action_reconstructed, _, _ = moral_model.bias_by_embs(embs_reconstructed.data.cpu().numpy())

        print(message, ";", round(bias_with_action[0].item(), 6), ";", round(bias_with_action_reconstructed.item(), 6))
    input("any key to continue")

    sentences_calc_moral = list(sentences_calc_moral_neg.keys())
    print("--" * 42)
    print("Atomic actions neg", len(sentences_calc_moral))
    print("--" * 42)
    for i, message in enumerate(sentences_calc_moral):
        break
        bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
        embs = torch.FloatTensor(embs)
        embs = embs.cuda()
        hidden = encoder(embs)
        embs_reconstructed = decoder(hidden)
        bias_with_action_reconstructed, _, _ = moral_model.bias_by_embs(embs_reconstructed.data.cpu().numpy())

        print(message, ";", round(bias_with_action[0].item(), 6), ";", round(bias_with_action_reconstructed.item(), 6))
    input("any key to continue")

    # start with eval atomic actions
    sentences_calc_moral = dataMoral.adaptbias_get_sentences_calc_moral_projections()
    sentences_calc_moral = list(sentences_calc_moral.keys())
    print("--" * 42)
    print("Atomic actions", len(sentences_calc_moral))
    print("--" * 42)
    for i, message in enumerate(sentences_calc_moral):
        break
        bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
        embs = torch.FloatTensor(embs)
        embs = embs.cuda()
        hidden = encoder(embs)
        embs_reconstructed = decoder(hidden)
        bias_with_action_reconstructed, _, _ = moral_model.bias_by_embs(embs_reconstructed.data.cpu().numpy())

        print(i, bias_with_action[1], bias_with_action[0], bias_with_action_reconstructed)


    sentences_calc_moral = dataMoral.adaptbias_get_sentences_quests_with_context_actions()
    sentences_calc_moral = list(sentences_calc_moral.keys())
    print("--" * 42)
    print("Context based actions", len(sentences_calc_moral))
    print("--" * 42)
    for i, message in enumerate(sentences_calc_moral):
        break
        bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
        embs = torch.FloatTensor(embs)
        embs = embs.cuda()
        hidden = encoder(embs)
        embs_reconstructed = decoder(hidden)
        bias_with_action_reconstructed, _, _ = moral_model.bias_by_embs(embs_reconstructed.data.cpu().numpy())

        print(i, bias_with_action[1], bias_with_action[0], bias_with_action_reconstructed)

    print("--" * 42)
    print("User study actions")
    print("--" * 42)
    norm_score, norm_rec_score, res, labels = user_study_bias_score(encoder, decoder)

    print("Computer norm values", norm_score, norm_rec_score)
    for a, b, br in res:
        #print(a, "User study bias", labels[a], "moral score:", round(b, 2), "moral score rec:", round(br, 2))
        print(a, ";", round(labels[a], 6), ";", round(b, 6), ";", round(br, 6))

    # sentence_context_moral = dataMoral.adaptbias_get_sentences_quests_with_actions()
    # res += flatten([a for a in [tmp[key] for key in list(tmp.keys())]])


def user_study_bias_score_(transormer_model='bert-large-nli-mean-tokens', moral_model=None):
    with torch.no_grad():
        if moral_model is None:
            moral_model = BERTSentenceSubspace(pca_framework='torch', transormer_model=transormer_model)

        sentences_calc_moral, labels = dataMoral.adaptbias_get_actions_adapt_moral()
        sentences_calc_moral = list(sentences_calc_moral.keys())
        user_study_bias = list()
        for message in tqdm(sentences_calc_moral):
            bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
            user_study_bias.append((bias_with_action[1],
                                    bias_with_action[0]))
            # print(bias_with_action[1], bias_with_action[0], bias_with_action_reconstructed)

        actions, bias = list(zip(*user_study_bias))
        bias = np.array(bias)
        norm_score = np.max(np.abs(bias))
        bias /= norm_score
        return norm_score, zip(actions, bias), labels


def user_study_bias_score(encoder, decoder, transormer_model='bert-large-nli-mean-tokens', moral_model=None):
    with torch.no_grad():
        #moral_model = BERTSentenceSubspace()
        #moral_model_torch = BERTSentenceSubspace(pca_framework='torch')
        if moral_model is None:
            moral_model = BERTSentenceSubspace(pca_framework='torch', transormer_model=transormer_model)
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            encoder.eval()
            decoder.eval()

        sentences_calc_moral, labels = dataMoral.adaptbias_get_actions_adapt_moral()
        sentences_calc_moral = list(sentences_calc_moral.keys())
        #sentences_calc_moral = ["harm"]
        user_study_bias = list()
        for message in tqdm(sentences_calc_moral):
            bias_with_action, embs, _ = moral_model.bias(message=message, return_emb="all")
            if not isinstance(embs, torch.FloatTensor):
                embs = torch.FloatTensor(embs)
            embs = embs.cuda()
            hidden = encoder(embs)
            embs_reconstructed = decoder(hidden)
            bias_with_action_reconstructed, _, _ = moral_model.bias_by_embs(
                embs_reconstructed.data.cpu().unsqueeze(dim=0), dim=1)
            if isinstance(bias_with_action_reconstructed, torch.Tensor):
                bias_with_action_reconstructed = bias_with_action_reconstructed.data.cpu().numpy().item()
            user_study_bias.append((bias_with_action[1],
                                    bias_with_action[0],
                                    bias_with_action_reconstructed))
            # print(bias_with_action[1], bias_with_action[0], bias_with_action_reconstructed)

        actions, bias, bias_rec = list(zip(*user_study_bias))
        bias = np.array(bias)
        norm_score = np.max(np.abs(bias))
        bias /= norm_score
        bias_rec = np.array(bias_rec)
        norm_rec_score = np.max(np.abs(bias_rec))
        bias_rec /= norm_rec_score
        return norm_score, norm_rec_score, zip(actions, bias, bias_rec), labels


def user_study_bias_score_with_precomputed_embeddings(encoder, decoder, sent2emb, verbose=True):
    with torch.no_grad():
        moral_model = BERTSentenceSubspace()
        moral_model_torch = BERTSentenceSubspace(pca_framework='torch')

        sentences_calc_moral, labels = dataMoral.adaptbias_get_actions_adapt_moral()
        sentences_calc_moral_keys = list(sentences_calc_moral.keys())

        user_study_bias = list()
        iter = sentences_calc_moral_keys
        if verbose:
            iter = tqdm(sentences_calc_moral_keys)
        for message in iter:
            sentences_calc_moral_emb = [sent2emb[s] for s in sentences_calc_moral[message]]
            bias_with_action, embs, _ = moral_model.bias_by_embs(sentences_calc_moral_emb)
            embs = torch.FloatTensor(sentences_calc_moral_emb)
            embs = embs.cuda()
            hidden = encoder(embs)
            embs_reconstructed = decoder(hidden)
            bias_with_action_reconstructed, _, _ = moral_model_torch.bias_by_embs(
                embs_reconstructed.data.cpu().unsqueeze(dim=0), dim=1)
            user_study_bias.append((message,
                                    bias_with_action,
                                    bias_with_action_reconstructed.data.cpu().numpy().item()))
            # print(bias_with_action[1], bias_with_action[0], bias_with_action_reconstructed)

        actions, bias, bias_rec = list(zip(*user_study_bias))
        bias = np.array(bias)
        norm_score = np.max(np.abs(bias))
        bias /= norm_score
        bias_rec = np.array(bias_rec)
        norm_rec_score = np.max(np.abs(bias_rec))
        bias_rec /= norm_rec_score
        return norm_score, norm_rec_score, zip(actions, bias, bias_rec), labels

def main():
    print('Generating emb...')
    model_name = 'bert-large-nli-mean-tokens'
    eval_model = './mort/adaptBias/results/{}/bert_model_adapted_run1/best_model.pt'.format(model_name)

    eval_model_path = eval_model.replace("best_model", "autoencoder")
    eval_model_path = eval_model.replace("best_model", "adapted_best_model")
    checkpoint = torch.load(eval_model_path, map_location=lambda storage, loc: storage.cuda(hp.gpu))
    hp_loaded_model = checkpoint['hp']

    if hp.gpu:
        cuda.set_device(hp_loaded_model.gpu)
    torch.manual_seed(hp_loaded_model.seed)

    encoder = model.Encoder(hp_loaded_model.emb_size, hp_loaded_model.hidden_size, dropout_rate=0.)
    decoder = model.Decoder(hp_loaded_model.hidden_size, hp_loaded_model.emb_size, dropout_rate=0.)

    emb = BERTSentence(device=hp_loaded_model.device)
    tmp_all_sentences_path = './data/adaptBias/tmp/sent2emb_{}.p'.format(emb.transormer_model)
    sent2emb = pickle.load(open(tmp_all_sentences_path, "rb"))

    if hp.gpu >= 0:
        encoder.cuda()
        decoder.cuda()

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    eval_autoencoder(encoder, decoder, sent2emb)

    """
    w2v = de_biassing_emb(encoder)
    eval_bias_analogy(w2v)

    print('Saving emb...')
    debias_emb_txt = 'src/debiased_{}/gender_debiased.txt'.format(sys.argv[1])
    debias_emb_bin = 'src/debiased_{}/gender_debiased.bin'.format(sys.argv[1])
    w2v.save_word2vec_format(debias_emb_bin, binary=True)
    w2v.save_word2vec_format(debias_emb_txt, binary=False)
    """


if __name__ == "__main__":
    main()
