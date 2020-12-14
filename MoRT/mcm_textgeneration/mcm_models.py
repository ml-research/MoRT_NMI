# coding=utf-8
# adapted from https://github.com/huggingface/transformers/tree/master/src/transformers/modeling_utils.py
import transformers
from transformers import GPT2LMHeadModel
import torch
import warnings
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.file_utils import add_start_docstrings_to_callable
# dirty hack to add mort to path
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../"))
from mort.funcs_mcm import BERTSentenceSubspace, MoRTSentenceSubspace
# for windows
# import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length if `past` is None else 1
            Indices of input sequence tokens in the vocabulary.
            If using `past` as an input make sure that `input_ids` are those of the last position.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__

        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`, defaults to :obj:`None`):
            `input_ids_length` = `sequence_length if `past` is None else 1
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            If using `past` as an input make sure that `token_type_ids` correspond to the `input_ids` of the last position.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


class GPT2MCMLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # by default, do not use the mcm until setup is performed
        self.use_mcm = False
        self.mcm = None
        self.mcm_tokenizer = None
        self.mcm_threshold = 0
        self.min_token_number = 1
        self.save_edge_cases = False

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past, "use_cache": kwargs["use_cache"]}

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def setup_mcm(self, device="cpu", transformer_model='bert-large-nli-mean-tokens',
                  mcm_tokenizer=None, threshold=0., use_mort=False, min_token_number=1,
                  save_edge_cases=False, file_descriptor="", use_mcm=True, working_path=None):
        print("Running setup_mcm")
        self.mcm_tokenizer = mcm_tokenizer
        self.mcm_threshold = threshold
        self.min_token_number = min_token_number
        self.save_edge_cases = save_edge_cases
        self.file_descriptor = file_descriptor
        if transformer_model == "pytorch-sentiment-analysis":
            sentiment_model = BERTGRUSentiment(transformers.BertModel.from_pretrained('bert-base-uncased'), 256, 1, 2, True, 0.25).to(torch.device(device))
            # expects model to be in same folder, can currently be found on server: /home/ml-stud-jhonermann/tut6-model.pt
            sentiment_model.load_state_dict(torch.load('tut6-model.pt'))
            sentiment_model.eval()
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

            def predict_sentiment_(sentence):
                tokens = tokenizer.tokenize(sentence)
                tokens = tokens[:tokenizer.max_model_input_sizes['bert-base-uncased'] - 2]
                indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
                tensor = torch.LongTensor(indexed).to(torch.device(device))
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(sentiment_model(tensor))
                return [[prediction.item(), sentence], []]

            self.mcm = predict_sentiment_
        else:
            # MoRT currently requires to change line 226 in mort/funcs_mcm.py to bias_with_action[0]))
            # and line 999 in mort/dataMoral.py to data = np.genfromtxt('../data/parsed_yes_no_BERTBias.csv', skip_header=1, dtype=str, delimiter=",")
            if use_mort:
                mcm_model = MoRTSentenceSubspace(device=device,
                                                 transormer_model=transformer_model,
                                                 filename_pickled_cluster=None,
                                                 pca_framework='sklearn',
                                                 eval_model_path=os.path.join(os.getcwd(),
                                                                              "../",
                                                                              "mort",
                                                                              "adaptBias",
                                                                              "results",
                                                                              transformer_model,
                                                                              "bert_model_adapted_run1",
                                                                              "adapted_best_model.pt"))

                def mcm_(query):
                    return mcm_model.bias(query, norm=mcm_model.norm, qa_template=False)

                self.mcm = mcm_
            else:
                mcm_model = BERTSentenceSubspace(device=device,
                                                 transormer_model=transformer_model,
                                                 filename_pickled_cluster=None,
                                                 pca_framework='sklearn', working_path=working_path)

                def mcm_(query):
                    return mcm_model.bias(query, norm=8.946814, qa_template=False)

                self.mcm = mcm_

        self.use_mcm = use_mcm

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, input_ids=input_ids, mcm=self.mcm,
                                                             use_mcm=self.use_mcm, tokenizer=self.mcm_tokenizer,
                                                             top_k=top_k, top_p=top_p, threshold=self.mcm_threshold,
                                                             mcm_keep_at_least=self.min_token_number,
                                                             edge_cases_file=self.file_descriptor if self.save_edge_cases else "")
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    input_ids: Tensor = None,
    use_mcm: bool = False,
    tokenizer=None,
    mcm=None,
    threshold: float = 0.0,
    mcm_keep_at_least: int = 1,
    edge_cases_file: str = "",
    eot_token: str = '',
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    if use_mcm:
        tokenlist = list(torch.nonzero(F.softmax(logit, dim=-1), as_tuple=True)[-1].tolist() for logit in logits)
        indices_to_remove = torch.ones(logits.size(), dtype=torch.bool)
        for next_tokens, current, sequence in zip(tokenlist, input_ids, range(logits.size()[0])):
            # decode tokens for mcm and retrieve scoring
            texts = tokenizer.batch_decode(torch.cat([torch.tensor([list(current)] * len(next_tokens)).to(input_ids.device),
                                                      torch.reshape(torch.tensor(next_tokens), (-1, 1)).to(input_ids.device)],
                                                     dim=-1),
                                           clean_up_tokenization_spaces=True)
            texts = [text.replace(eot_token, '') for text in texts]
            mcm_res, _, _ = mcm(texts)
            index_scores = list(zip(mcm_res[0], next_tokens, texts))
            index_scores.sort(reverse=True) # TODO index_scores sorting inpalce ?
            # try to only keep positive ratings
            keep = [entry[1] for entry in index_scores if entry[0] > threshold]
            # if there is no option left retrieve the best rated n indices
            if len(keep) <= mcm_keep_at_least:
                if len(index_scores) > mcm_keep_at_least:
                    keep = [entry[1] for entry in index_scores[:mcm_keep_at_least]]
                else:
                    keep = [entry[1] for entry in index_scores[:]]
            # consider the step an edge case if some values are filtered, but number of remaining tokens is still above the keep-threshold
            elif len(edge_cases_file) > 0 and len(keep) != len(index_scores):
                with open("edge_cases_{}.txt".format(edge_cases_file), "a", encoding="utf-8") as file:
                    for index_score in index_scores:
                        file.write("{}|{}\n".format(index_score[0], index_score[2]))
            for i in keep:
                indices_to_remove[sequence][i] = 0
        logits[indices_to_remove] = filter_value
    return logits


class DialoGPT2MCMLMHeadModel(GPT2MCMLMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def setup_mcm(self, device="cpu", transformer_model='bert-large-nli-mean-tokens',
                  mcm_tokenizer=None, threshold=0, use_mort=False, min_token_number=1,
                  save_edge_cases=False, file_descriptor="", use_mcm=True, start_mcm_token=None, num_text_splits=1):
        super().setup_mcm(device=device, transformer_model=transformer_model,
                          mcm_tokenizer=mcm_tokenizer, threshold=threshold, use_mort=use_mort, min_token_number=min_token_number,
                          save_edge_cases=save_edge_cases, file_descriptor=file_descriptor, use_mcm=use_mcm)

        self.start_mcm_token = start_mcm_token
        self.num_text_splits = num_text_splits

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering_dialogpt(scores, input_ids=input_ids, mcm=self.mcm,
                                                                      use_mcm=self.use_mcm, tokenizer=self.mcm_tokenizer,
                                                                      top_k=top_k, top_p=top_p, threshold=self.mcm_threshold,
                                                                      start_mcm_token=self.start_mcm_token,
                                                                      num_text_splits=self.num_text_splits,
                                                                      mcm_keep_at_least=self.min_token_number,
                                                                      edge_cases_file=self.file_descriptor if self.save_edge_cases else "")
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids


def top_k_top_p_filtering_dialogpt(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    input_ids: Tensor = None,
    use_mcm: bool = False,
    tokenizer=None,
    mcm=None,
    threshold: float = 0.0,
    mcm_keep_at_least: int = 1,
    edge_cases_file: str = "",
    start_mcm_token: str = None,
    num_text_splits: int = 1
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    if use_mcm:
        tokenlist = list(torch.nonzero(F.softmax(logit, dim=-1), as_tuple=True)[-1].tolist() for logit in logits)
        indices_to_remove = torch.ones(logits.size(), dtype=torch.bool)
        for next_tokens, current, sequence in zip(tokenlist, input_ids, range(logits.size()[0])):
            # each sequence is processed seperately
            #index_scores = []
            texts = []
            for next_token in next_tokens: # TODO impl this parallel with python multiprocessing
                # decode tokens for mcm and retrieve scoring
                text = tokenizer.decode(torch.cat([current, torch.tensor([next_token]).to(input_ids.device)], dim=-1),
                                        clean_up_tokenization_spaces=True)
                if start_mcm_token:
                    text_splits = text.split(start_mcm_token)[-num_text_splits:]
                    text = ' '.join(text_splits)
                texts.append(text)
                #index_scores.append((mcm(text)[0][0], next_token, text))
            mcm_res, _, _ = mcm(texts)
            index_scores = list(zip(mcm_res[0], next_tokens, texts))
            index_scores.sort(reverse=True)
            # try to only keep positive ratings
            keep = [entry[1] for entry in index_scores if entry[0] > threshold]
            # if there is no option left retrieve the best rated n indices
            if len(keep) <= mcm_keep_at_least:
                if len(index_scores) > mcm_keep_at_least:
                    keep = [entry[1] for entry in index_scores[:mcm_keep_at_least]]
                else:
                    keep = [entry[1] for entry in index_scores[:]]
            # consider the step an edge case if some values are filtered, but number of remaining tokens is still above the keep-threshold
            elif len(edge_cases_file) > 0 and len(keep) != len(index_scores):
                with open("edge_cases_{}.txt".format(edge_cases_file), "a", encoding="utf-8") as file:
                    for index_score in index_scores:
                        file.write("{}|{}\n".format(index_score[0], index_score[2]))
            indices_to_remove[sequence][keep] = 0
        logits[indices_to_remove] = filter_value
    return logits


def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


class OpenAIGPTMCMLMHeadModel(transformers.OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # by default, do not use the mcm until setup is performed
        self.use_mcm = False
        self.mcm = None
        self.mcm_tokenizer = None
        self.mcm_threshold = 0
        self.min_token_number = 1
        self.save_edge_cases = False

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

        # return transformers.CausalLMOutput(
        #    loss=loss,
        #    logits=lm_logits,
        #    hidden_states=transformer_outputs.hidden_states,
        #    attentions=transformer_outputs.attentions,
        #)

    def setup_mcm(self, device="cpu", transformer_model='bert-large-nli-mean-tokens',
                  mcm_tokenizer=None, threshold=0, use_mort=False, min_token_number=1, save_edge_cases=False, file_descriptor=""):
        self.mcm_tokenizer = mcm_tokenizer
        self.mcm_threshold = threshold
        self.min_token_number = min_token_number
        self.save_edge_cases = save_edge_cases
        self.file_descriptor = file_descriptor
        if transformer_model == "pytorch-sentiment-analysis":
            sentiment_model = BERTGRUSentiment(transformers.BertModel.from_pretrained('bert-base-uncased'), 256, 1, 2, True, 0.25).to(torch.device(device))
            # expects model to be in same folder, can currently be found on server: /home/ml-stud-jhonermann/tut6-model.pt
            sentiment_model.load_state_dict(torch.load('tut6-model.pt'))
            sentiment_model.eval()
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

            def predict_sentiment_(sentence):
                tokens = tokenizer.tokenize(sentence)
                tokens = tokens[:tokenizer.max_model_input_sizes['bert-base-uncased'] - 2]
                indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
                tensor = torch.LongTensor(indexed).to(torch.device(device))
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(sentiment_model(tensor))
                return [[prediction.item(), sentence], []]

            self.mcm = predict_sentiment_
        else:
            # MoRT currently requires to change line 226 in mort/funcs_mcm.py to bias_with_action[0]))
            # and line 999 in mort/dataMoral.py to data = np.genfromtxt('../data/parsed_yes_no_BERTBias.csv', skip_header=1, dtype=str, delimiter=",")
            if use_mort:
                mcm_model = MoRTSentenceSubspace(device=device,
                                                 transormer_model=transformer_model,
                                                 filename_pickled_cluster=None,
                                                 pca_framework='sklearn',
                                                 eval_model_path=os.path.join(os.getcwd(),
                                                                              "../",
                                                                              "mort",
                                                                              "adaptBias",
                                                                              "results",
                                                                              transformer_model,
                                                                              "bert_model_adapted_run1",
                                                                              "adapted_best_model.pt"))

                def mcm_(query):
                    return mcm_model.bias(query, norm=mcm_model.norm, qa_template=False)

                self.mcm = mcm_
            else:
                mcm_model = BERTSentenceSubspace(device=device,
                                                 transormer_model=transformer_model,
                                                 filename_pickled_cluster=None,
                                                 pca_framework='sklearn')

                def mcm_(query):
                    return mcm_model.bias(query, norm=8.946814, qa_template=False)

                self.mcm = mcm_

        self.use_mcm = True

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, input_ids=input_ids, mcm=self.mcm,
                                                             use_mcm=self.use_mcm, tokenizer=self.mcm_tokenizer,
                                                             top_k=top_k, top_p=top_p, threshold=self.mcm_threshold,
                                                             mcm_keep_at_least=self.min_token_number,
                                                             edge_cases_file=self.file_descriptor if self.save_edge_cases else "")
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids
