from pathlib import Path
from typing import Union, List
import sys
import os

mort_path = '/workspace/repositories/MoRT'
if not os.path.isdir(mort_path):
    mort_path = '../MoRT'

sys.path.append(mort_path)

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2PreTrainedModel
from mcm_textgeneration.mcm_models import GPT2MCMLMHeadModel
from mcm_textgeneration.mcm_models import top_k_top_p_filtering
from transformers.generation_utils import calc_banned_bad_words_ids

from utils import utils

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 < max_sequence_length:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


# TODO: convert to HuggingFace pipeline
class GPT2Generation:
    STOP_TOKEN = "<|endoftext|>"

    def __init__(self, model: Union[str, Path, GPT2PreTrainedModel] = 'gpt2', tokenizer: str = 'gpt2', seed: int = 42,
                 threshold: float = 0.0, k=0, p=0.9,min_token_number=30, use_mcm=False):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        # Set up model
        if isinstance(model, Path) or isinstance(model, str):
            model = GPT2MCMLMHeadModel.from_pretrained(str(model))
        model.setup_mcm(device=self.device,
                        transformer_model='bert-large-nli-mean-tokens',
                        mcm_tokenizer=tokenizer, threshold=threshold,
                        use_mort=False,
                        save_edge_cases=False,
                        file_descriptor='',
                        min_token_number=min_token_number,
                        use_mcm=use_mcm, working_path=mort_path)
        self.model = model.to(self.device)

        # Set up tokenizer
        # IMPORTANT: Note that setting the pad token like this in the constructor gives the pad_token the
        # pad_token_id = 50256, which normally belongs to the <EOS> token_id in GPT2. This is a very ugly
        # way that works at the moment of setting the pad_token_id to the <EOS> token that is already
        # included in the vocab size.
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token=self.STOP_TOKEN)
        self.k = k
        self.p = p
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<GPT2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 temperature: float = 1.0,
                 bad_words_ids: List[List[int]] = None,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, pad_to_max_length=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):
                logits, past = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                          **model_kwargs)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = logits[:, -1, :]

                if bad_words_ids is not None:
                    # calculate a list of banned tokens according to bad words
                    banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                    # TODO: use a vectorized operation
                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    # Top-p/top-k filtering
                    next_token_logits = top_k_top_p_filtering(next_token_logits, input_ids=input_ids,
                                                              #top_k=max(self.k, 50),
                                                              top_k=max(self.k, 10),
                                                              top_p=self.p,
                                                              use_mcm=self.model.use_mcm,
                                                              tokenizer=self.tokenizer,
                                                              mcm=self.model.mcm,
                                                              threshold=self.model.mcm_threshold,
                                                              mcm_keep_at_least=self.model.min_token_number,
                                                              edge_cases_file="",
                                                              eot_token=self.STOP_TOKEN)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs

    def generate_multiple(self,
                          prompt: str,
                          max_len: int = 20,
                          temperature: float = 1.0,
                          k: int = 0,
                          p: float = 0.9,
                          num_return_sequences: int = 1,
                          sample: bool = True,
                          repetition_penalty: float = 1.0):
        max_len = adjust_length_to_model(max_len, max_sequence_length=self.model.config.max_position_embeddings)

        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        prompt_len = len(encoded_prompt[0])

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_len + prompt_len,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=sample,
            num_return_sequences=num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        decoded_outputs = []
        for output in output_sequences:
            output = output[prompt_len:]
            try:
                stop_index = [i for i, x in enumerate(output) if x == self.tokenizer.eos_token_id][0]
            except IndexError:
                stop_index = None
            output = output[:stop_index]
            decoded_outputs.append(self.tokenizer.decode(output, clean_up_tokenization_spaces=True))

        return decoded_outputs


def test_generate():
    generator = GPT2Generation()
    prompt = [
        '<|endoftext|>in this paper we',
        '<|endoftext|>we are trying to',
        '<|endoftext|>The purpose of this workshop is to check whether we can'
    ]
    out = generator.generate(prompt)
    print(*out, sep='\n')


def test_generate_multiple():
    generator = GPT2Generation()
    prompt = 'in this paper we'
    out = generator.generate_multiple(prompt)
    print(*out, sep='\n')
