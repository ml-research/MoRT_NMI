import os

_working_path = os.environ['PYTHONPATH'].split(':')[0]


class Hyperparams_rec:
    model_name = 'bert-large-nli-mean-tokens'
    run_id = "rec"
    save_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/'.format(model_name, run_id))
    eval_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/adapted_best_model.pt'.format(model_name,
                                                                                                     run_id))

    path_explanations = os.path.join(_working_path, 'mort/adaptBias/data/interactive/test.json')

    # pre-train autoencoder
    pre_train_autoencoder = True

    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 3000

    # pre-train classifier
    pre_train_classifier = False  # TODO
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 4000
    sampling = 'over_sampling'  # over_sampling or under_sampling

    emb_binary = False
    batch_size = 32
    learning_rate = 0.1
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True

    gender_no_gender_loss = False  # TODO
    classifier_loss = False  # TODO

    max_grad_norm = None
    moral_vektor_loss = True
    cossim_loss = False
    reconstruction_loss = True

    moral_loss_rate = 0.0001  # TODO
    inmoral_loss_rate = 0.0001  # TODO
    gender_stereotype_loss_rate = 0.0001  # TODO
    gender_no_gender_loss_rate = 0.0001  # TODO

    decoder_loss_rate = 1.0
    moral_vektor_loss_rate = 0.1
    cossim_loss_rate = 0.1

    emb_size = 1024
    hidden_size = 1024

    epochs = 1000
    seed = 0
    gpu = 0

    if gpu >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # mcm
    mcm_num_template_questions = 10

class Hyperparams_study:
    model_name = 'bert-large-nli-mean-tokens'
    run_id = "user_studyX"
    save_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/'.format(model_name, run_id))
    eval_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/adapted_best_model.pt'.format(model_name,
                                                                                                     run_id))

    path_explanations = os.path.join(_working_path, 'mort/adaptBias/data/interactive/test.json')

    # pre-train autoencoder
    # pre_train_autoencoder = True
    # pre_train_autoencoder = save_model + 'autoencoder.pt'
    pre_train_autoencoder = os.path.join(_working_path,
                                         'mort/adaptBias/results/{}_{}/bert_model/{}'.format(model_name,
                                                                                             'user_study',
                                                                                             'adapted_best_model.pt'))
    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 3000

    # pre-train classifier
    pre_train_classifier = False  # TODO
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 4000
    sampling = 'over_sampling'  # over_sampling or under_sampling

    emb_binary = False
    batch_size = 32
    learning_rate = 0.1
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True

    gender_no_gender_loss = False  # TODO
    classifier_loss = False  # TODO

    max_grad_norm = None
    moral_vektor_loss = True
    cossim_loss = False
    reconstruction_loss = True

    moral_loss_rate = 0.0001  # TODO
    inmoral_loss_rate = 0.0001  # TODO
    gender_stereotype_loss_rate = 0.0001  # TODO
    gender_no_gender_loss_rate = 0.0001  # TODO

    decoder_loss_rate = 1.0
    moral_vektor_loss_rate = 0.1
    cossim_loss_rate = 0.1

    emb_size = 1024
    hidden_size = 1024

    epochs = 200
    seed = 0
    gpu = 0

    if gpu >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # mcm
    mcm_num_template_questions = 10


class Hyperparams_interactive:
    model_name = 'bert-large-nli-mean-tokens'
    run_id = "explanation_test_runX_tasksjointly"
    save_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/'.format(model_name, run_id))
    eval_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/adapted_best_model.pt'.format(model_name,
                                                                                                     run_id))

    path_explanations = os.path.join(_working_path, 'mort/adaptBias/data/interactive/test.json')

    # pre-train autoencoder
    # pre_train_autoencoder = True
    # pre_train_autoencoder = save_model + 'autoencoder.pt'
    pre_train_autoencoder = os.path.join(_working_path,
                                         'mort/adaptBias/results/{}_{}/bert_model/{}'.format(model_name,
                                                                                             'user_study',
                                                                                             'adapted_best_model.pt'))
    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 3000

    # pre-train classifier
    pre_train_classifier = False  # TODO
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 4950
    sampling = 'None'  # over_sampling or under_sampling

    emb_binary = False
    batch_size = 32
    learning_rate = 0.002
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True

    gender_no_gender_loss = False  # TODO
    classifier_loss = False  # TODO

    moral_vektor_loss = True
    cossim_loss = True
    max_grad_norm = None
    reconstruction_loss = True

    moral_loss_rate = 0.0001  # TODO
    inmoral_loss_rate = 0.0001  # TODO
    gender_stereotype_loss_rate = 0.0001  # TODO
    gender_no_gender_loss_rate = 0.0001  # TODO

    decoder_loss_rate = 1.
    moral_vektor_loss_rate = 1.
    cossim_loss_rate = .1

    emb_size = 1024
    hidden_size = 1024

    epochs = 1000
    seed = 0
    gpu = 0

    if gpu >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # mcm
    mcm_num_template_questions = 10


class Hyperparams_interactive_interface:
    model_name = 'bert-large-nli-mean-tokens'
    run_id = "PLACEHOLDER"
    save_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/'.format(model_name, run_id))
    eval_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/adapted_best_model.pt'.format(model_name,
                                                                                                     run_id))

    path_explanations = None #os.path.join(_working_path, 'mort/adaptBias/data/interactive/{}.json')

    # pre-train autoencoder
    # pre_train_autoencoder = True
    # pre_train_autoencoder = save_model + 'autoencoder.pt'
    pre_train_autoencoder_placeholder = os.path.join(_working_path,
                                         'mort/adaptBias/results/{}_{}/bert_model/{}'.format(model_name,
                                                                                             '{}', '{}'))
    pre_train_autoencoder = None
    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 3000

    # pre-train classifier
    pre_train_classifier = False  # TODO
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 4950
    sampling = 'None'  # over_sampling or under_sampling

    emb_binary = False
    batch_size = 32
    learning_rate = 0.002
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True

    gender_no_gender_loss = False  # TODO
    classifier_loss = False  # TODO

    moral_vektor_loss = False
    cossim_loss = True
    max_grad_norm = None
    reconstruction_loss = True

    moral_loss_rate = 0.0001  # TODO
    inmoral_loss_rate = 0.0001  # TODO
    gender_stereotype_loss_rate = 0.0001  # TODO
    gender_no_gender_loss_rate = 0.0001  # TODO

    decoder_loss_rate = .001
    moral_vektor_loss_rate = .001
    cossim_loss_rate = 1.

    emb_size = 1024
    hidden_size = 1024

    epochs = 100
    seed = 0
    gpu = 0

    if gpu >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # mcm
    mcm_num_template_questions = 10

# run 3,4,5
class Hyperparams_interactive_works_old:
    model_name = 'bert-large-nli-mean-tokens'
    run_id = "explanation_test_run5"
    save_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/'.format(model_name, run_id))
    eval_model = os.path.join(_working_path,
                              'mort/adaptBias/results/{}_{}/bert_model/adapted_best_model.pt'.format(model_name,
                                                                                                     run_id))

    path_explanations = os.path.join(_working_path, 'mort/adaptBias/data/interactive/test.json')

    # pre-train autoencoder
    # pre_train_autoencoder = True
    # pre_train_autoencoder = save_model + 'autoencoder.pt'
    pre_train_autoencoder = os.path.join(_working_path,
                                         'mort/adaptBias/results/{}_{}/bert_model/{}'.format(model_name,
                                                                                             'user_study',
                                                                                             'adapted_best_model.pt'))
    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 3000

    # pre-train classifier
    pre_train_classifier = False  # TODO
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 4950
    sampling = 'None'  # over_sampling or under_sampling

    emb_binary = False
    batch_size = 32
    learning_rate = 0.002
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True

    gender_no_gender_loss = False  # TODO
    classifier_loss = False  # TODO

    moral_vektor_loss = False
    cossim_loss = True
    max_grad_norm = None
    reconstruction_loss = False

    moral_loss_rate = 0.0001  # TODO
    inmoral_loss_rate = 0.0001  # TODO
    gender_stereotype_loss_rate = 0.0001  # TODO
    gender_no_gender_loss_rate = 0.0001  # TODO

    decoder_loss_rate = .001
    moral_vektor_loss_rate = .001
    cossim_loss_rate = 1.

    emb_size = 1024
    hidden_size = 1024

    epochs = 1000
    seed = 0
    gpu = 0

    if gpu >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # mcm
    mcm_num_template_questions = 10
