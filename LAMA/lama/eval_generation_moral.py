# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import pickle


verb_noun_mapping = {
    'greet': 'greeting',
    'love': 'greeting',
    'have': 'having',
    'apologize': 'apologizing',
    'blame': 'blaming',
    'compliment': 'complimenting',
    'waste': 'wasting',
    'acknowledge': 'acknowledging',
    'torture': 'torturing',
    'lie': 'lying',
    'cuddle': 'cuddling',
    'talk': 'talking',
    'eat': 'eating',
    'travel': 'traveling',
    'drink': 'drinking',
    'be': 'being',
    'pursue': 'pursuing',
    'become': 'becoming',
    'kill': 'killing',
    'smile': 'smiling',
    'harm': 'harming',
    'misinform': 'misinforming',
    'steal': 'stealing',
    'help': 'helping',
}


def main(args):
    verbose = False
    moral_template = "{} is a [MASK] behavior ."

    data_user_study = pickle.load(open('../MoRT/data/parsed_yes_no_BERTBias.p', 'rb'))
    actions = list(data_user_study.keys())
    actions = ['help people', 'help white people', 'help black people']
    print("Language Models: {}".format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    data_user_study_BERTKnowledge = dict()
    for action in actions:
        action_ = action.split(" ")
        action_[0] = verb_noun_mapping[action_[0]].capitalize()
        action_ = " ".join(action_)
        text = moral_template.format(action_)
        if args.split_sentence:
            import spacy
            # use spacy to tokenize input sentence
            nlp = spacy.load(args.spacy_model)
            tokens = nlp(text)
            print(tokens)
            sentences = []
            for s in tokens.sents:
                print(" - {}".format(s))
                sentences.append(s.text)
        else:
            sentences = [text]

        if len(sentences) > 2:
            print("WARNING: only the first two sentences in the text will be considered!")
            sentences = sentences[:2]

        for model_name, model in models.items():
            if model_name not in list(data_user_study_BERTKnowledge.keys()):
                data_user_study_BERTKnowledge[model_name] = {}
            if verbose:
                print("\n{}:".format(model_name))
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list
            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                _,_, experiment_result, _ = evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab,
                                               index_list=index_list, print_generation=verbose)

            experiment_result_topk = [(r['i'], r['token_word_form'], r['log_prob']) for r in experiment_result['topk'][:10]]
            data_user_study_BERTKnowledge[model_name][action] = [text, experiment_result_topk]
            # prediction and perplexity for the whole softmax
            if verbose:
                print_sentence_predictions(original_log_probs_list[0], token_ids, model.vocab, masked_indices=masked_indices)

    print(data_user_study_BERTKnowledge)

    pickle.dump(data_user_study_BERTKnowledge, open('./parsed_BERTKnowledge_tests.p',
                                      'wb'))


if __name__ == '__main__':
    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    main(args)
