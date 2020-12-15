import torch
from sklearn.decomposition import PCA
from mort.funcs_mcm import get_sen_embedding_from, BERTSentenceSubspace, BERTSentence, pytorch_pca_transform


def compute_PCA(n_components, x):
    reduc = PCA(n_components=n_components, svd_solver='full')
    res = reduc.fit_transform(x)
    return res, reduc


def get_model_config(model_type, transformer_model='bert-large-nli-mean-tokens'):
    adapted = True
    if model_type == 'orig':
        adapted = False
        transformer_model_adapted = transformer_model
        adapted_model_name = ''
    elif model_type == 'sembert':
        # seems to be capable handling negation and other semantic context
        adapted = False
        transformer_model = 'data/SentenceBERTmodels/SemBERT_new'
        transformer_model_adapted = transformer_model
        adapted_model_name = ''
    elif model_type == 'sembert-stsb':
        # sembert seems to give better results (first impression!)
        adapted = False
        transformer_model = 'data/SentenceBERTmodels/SemBERT_STSB_new'
        transformer_model_adapted = transformer_model
        adapted_model_name = ''
    elif model_type == 'rec':
        transformer_model_adapted = '{}_rec'.format(transformer_model)
        adapted_model_name = 'autoencoder.pt'
    elif model_type == 'study':
        transformer_model_adapted = '{}_user_study'.format(transformer_model)
        adapted_model_name = 'adapted_best_model.pt'
    elif 'xil_interface#' in model_type:
        type_ = model_type.split('#')
        type_ = type_[1]
        transformer_model_adapted = '{}{}'.format(
            transformer_model, type_)
        adapted_model_name = 'adapted_best_model.pt'
    elif model_type == 'xil_test':
        transformer_model_adapted = '{}_explanation_test_run1_tasksjointly'.format(
            transformer_model)
        adapted_model_name = 'adapted_best_model.pt'
    else:
        # transformer_model_adapted = '{}_explanation_test_run5'.format(transformer_model_) if adapted_ else transformer_model_
        # transformer_model_adapted = '{}_explanation_test_run_train_Tasksjointly'.format(transformer_model_) if adapted_ else transformer_model_
        raise ValueError('not found')

    eval_model_path = './mort/adaptBias/results/{}/bert_model/{}'.format(transformer_model_adapted, adapted_model_name)
    print('Using model at path:', '\n', eval_model_path)
    if adapted:
        transformer_model_adapted += '_{}'.format(adapted_model_name.replace('.pt', ''))

    return transformer_model, adapted, transformer_model_adapted, eval_model_path


def init_models(transormer_model, qa_template, eval_model_path, adapted=False, working_path=None,
                pca_framework='torch'):
    device = "cpu"
    filename_pickled_cluster = None

    subspace = True

    print("Start init MCM model")
    if adapted:
        # 7.9742536544799805 for rec
        raise ValueError("not included in this version")
    else:
        if subspace:
            if 'SemBERT_STSB' in transormer_model:
                norm = 11.730813
            elif 'SemBERT' in transormer_model:
                norm = 13.289515
            else:
                norm = 8.946814
            model_mcm = BERTSentenceSubspace(device=device,
                                             transormer_model=transormer_model,
                                             filename_pickled_cluster=filename_pickled_cluster,
                                             pca_framework=pca_framework,
                                             working_path=working_path)

            def bias(query, batch_size=10, show_progress_bar=False):
                return model_mcm.bias(query, norm=norm, qa_template=qa_template,
                                      batch_size=batch_size,
                                      show_progress_bar=show_progress_bar)
        else:
            model_mcm = BERTSentence(transormer_model=transormer_model)

            def bias(query, batch_size=10, show_progress_bar=False):
                res = model_mcm.bias(query,
                                     batch_size=batch_size,
                                     show_progress_bar=show_progress_bar)
                return res, None, None

    def mcm_get_bias_(query, batch_size=10, show_progress_bar=False):
        return bias(query, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def get_embedding_(query):
        return model_mcm.get_sen_embedding(query)

    print("Finished init MCM model")

    return mcm_get_bias_, get_embedding_, model_mcm