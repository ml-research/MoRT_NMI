import numpy as np
import argparse
from scipy import spatial
import scipy
from mort.explain.interactive_widget import build_explain_database, interactiveMCM
from mort.utils import init_models, get_model_config
from mort.adaptBias.train import run as run_xil_

parser = argparse.ArgumentParser(description='Crazy Stuff, Hard to Explain')


def run_xil(explanation_path, base_model):
    setting = 'xil_interface#'+explanation_path
    meta_info = run_xil_(setting, base_model=base_model)
    return meta_info


def run_explain(model_name, mcm_get_bias, qa_template, adapted=False, interactive=True, session=None):
    # get_sen_embedding_ = get_sen_embedding_from('bertsentence')
    action_embedding, distances, pos_2d = build_explain_database(model_name,
                                                                 mcm_get_bias,
                                                                 qa_template, adapted=adapted)

    #X_emb = np.array(action_embedding["embeddings"])
    # X_pca_emb = np.array(action_embedding["pca_embeddings"])

    verbs = np.array(action_embedding["verbs"])
    scores = np.array(action_embedding["scores"])

    # build distance matrix here
    data = action_embedding["embeddings"]

    num_samples = 1001
    data = data[:num_samples]
    verbs = verbs[:num_samples]
    scores = scores[:num_samples]

    num_sample_to_explain = 20

    # nbrs_pca_emb = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X_pca_emb)

    def compute_query(query):
        score, emb, pca_emb = mcm_get_bias(query)
        pca_emb = pca_emb.squeeze()
        if not isinstance(emb, np.ndarray):
            emb = emb.cpu().numpy()

        distances = scipy.spatial.distance.cdist([emb], data, "cosine")[0]
        indices = np.argsort(distances)
        distances, indices = distances[indices[:num_sample_to_explain]], indices[:num_sample_to_explain]
        # distances, indices = nbrs_emb.kneighbors([emb])
        print(" -- " * 21 + "Orginal Emb" + " -- " * 21)
        print("Score:", score)
        closest_verbs = verbs[indices]
        closest_verbs_scores = scores[indices]

        print(closest_verbs, distances)

        """
        for nn_actions in verbs[indices].squeeze():
            score, _, _ = mcm_get_bias([nn_actions])
            print("Action", nn_actions, "Score:", score)
        print(" -- " * 42)
        """

        return closest_verbs, score, closest_verbs_scores

    compute_query('init')
    model_base = 'bert-large-nli-mean-tokens'
    if model_name == model_base:
        base_model = ('rec', 'autoencoder.pt')
    else:
        base_model = (model_name.split('_')[1],
                      '{}.pt'.format('_'.join(model_name.split('_')[2:])))
        #'bert-large-nli-mean-tokens_XILinterface-1599137597.9287124-0_adapted_best_model'
    if interactive:
        pos_2d = pos_2d[:num_samples]
        MCM = interactiveMCM(query_func=compute_query, verbs=verbs,
                             pos=pos_2d, mcm_get_bias=mcm_get_bias,
                             qa_template=qa_template, session=session)
        result = MCM.run()
        if result['status'] == 'stop' or result['status'] == 'init':
            return False, None, result
        elif result['status'] == 'saved':
            model_path = run_xil(explanation_path=result['path'],
                                 base_model=base_model)
            return True, model_path, result
        else:
            print("Should not happen?")
            return False, None, None
    else:
        while True:
            query = input("Type an action \n")
            if query.lower() == "stop":
                break
            compute_query(query)
        return False, None, None


def run_mort(mcm_get_bias):
    while True:
        query = input("Type an action \n")

        if query.lower() == "stop":
            break

        score, emb, pca_emb = mcm_get_bias(query)
        # pca_emb = pca_emb.squeeze()
        print("Score:", score)


def _init_models(model_name, model_base, qa_template):
    transformer_model, adapted, transformer_model_adapted, eval_model_path = get_model_config(model_name,
                                                                                              transformer_model=model_base)
    mcm_get_bias, _, _ = init_models(transformer_model, qa_template,
                                     eval_model_path=eval_model_path,
                                     adapted=adapted)
    save_name = transformer_model_adapted

    return mcm_get_bias, adapted, save_name


def main(model_name, model_base, plot=True, qa_template=True, explain=False):
    # args = parser.parse_args()
    if explain:
        re_run = True
        session = None
        while re_run:
            mcm_get_bias, adapted, save_name = _init_models(model_name, model_base, qa_template)
            re_run, meta_info, result = run_explain(save_name, mcm_get_bias, qa_template,
                                            adapted=adapted,
                                            interactive=plot,
                                            session=session)
            if re_run:
                model_name = 'xil_interface#'+meta_info.split('/')[-3]
                model_name = model_name.replace(model_base, '')
                session = result['session']
        exit()
    else:
        mcm_get_bias, _, _ = _init_models(model_name, model_base, qa_template)
        run_mort(mcm_get_bias)
        exit()


def compute_norm_value():
    qa_template = True
    transformer_model, adapted, transformer_model_adapted, eval_model_path = get_model_config(args.model)
    mcm_get_bias, _, model = init_models(transformer_model, qa_template, eval_model_path=eval_model_path, adapted=adapted)
    from mort.adaptBias.eval import user_study_bias_score_
    norm_score, _, _ = user_study_bias_score_(transformer_model, moral_model=model)
    print(norm_score)
    exit()


if __name__ == '__main__':
    #transformer_model_, qa_template_ = 'data/SentenceBERTmodels/SemBERT', True # !!! case sensitive
    parser = argparse.ArgumentParser(description='XIL on MCM')
    parser.add_argument('--model', default=None, type=str,
                        #choices=['orig', 'rec', 'study', 'xil_test', 'sembert', 'sembert-stsb', 'xil_interface#_XILinterface-1599125294.4569185-0'],
                        help='', required=True)
    # ./mort/results/explain/bert-large-nli-mean-tokens_XILinterface-1599125294.4569185-0_adapted_best_model_adapted_questions

    parser.add_argument('--model-base', default='bert-large-nli-mean-tokens', type=str, choices=['bert-large-nli-mean-tokens',
                                                                         'roberta-large-nli-stsb-mean-tokens'],
                        help='')
    parser.add_argument('--qa-template', help="if not set the raw input is used without the question/answer template",
                    action="store_true")
    parser.add_argument('--explain', help="",
                        action="store_true")

    args = parser.parse_args()

    if args.model_base == 'roberta-large-nli-stsb-mean-tokens' and args.model != 'orig':
        raise ValueError('roberta only usable with orig. TODO adapt code')
    #compute_norm_value()
    main(args.model, plot=True, qa_template=args.qa_template, explain=args.explain, model_base=args.model_base)
