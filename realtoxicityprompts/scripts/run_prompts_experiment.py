import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm

from generation.generation import gpt2, gpt3, gpt2_affect, gpt2_ctrl, \
    openai_gpt, ctrl, pplm, gpt2mcm
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.utils import load_jsonl, batchify
import sys
import os


ALLOWED_MODELS = ['gpt3', 'gpt2', 'gpt2-affect', 'gpt2-ctrl', 'gpt2-greedy', 'gpt2-naughty-list',
                  'pplm', 'ctrl', 'openai-gpt', 'xlnet',
                  'gpt2mcm']


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if response['response']:
            response = unpack_scores(response['response'])[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]],
            output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--perspective-rate-limit', default=25)
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--start-index', default=0)
@click.option('--end-index', default=-1)
@click.option('--mcm-t', default=0.0)
@click.option('--mcm-keepmin', default=30)
@click.option('--top-k', default=50)
@click.option('--resume/--no-resume', default=False)
@click.option('--usemcm/--no-usemcm', default=True, help='Use mcm or not. When used with --model-type gpt2mcm')
def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, model_type: str, n: int,
         max_tokens: int, batch_size: int, perspective_rate_limit: int, start_index: int, end_index: int,
         mcm_t: float, mcm_keepmin: int, top_k: int,
         resume: bool, usemcm: bool):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        dataset = dataset[start_index:end_index]
        prompts = pd.json_normalize(dataset['prompt'])['text']
    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-naughty-list', 'pplm', 'gpt2mcm']:
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'gpt2-ctrl':
            prompts = pd.Series('<|nontoxic|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'xlnet':
            prompts = pd.Series('<s>').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'openai-gpt':
            prompts = pd.Series('. ').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')

    print('Prompts:', '\n', prompts)
    # Create output files
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=resume)
    generations_file = output_dir / 'generations.jsonl'
    perspective_file = output_dir / 'perspective.jsonl'
    output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=perspective_file,
                                    total=len(prompts) * n,
                                    rate_limit=perspective_rate_limit)

    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file)
    elif model_type == 'gpt2mcm':
        generations_iter = gpt2mcm(prompts=prompts,
                                   max_len=max_tokens,
                                   num_samples=n,
                                   batch_size=batch_size,
                                   model_name_or_path=model,
                                   out_file=generations_file,
                                   threshold=mcm_t,
                                   k=top_k,
                                   min_token_number=mcm_keepmin,
                                   use_mcm=usemcm)

    elif model_type == 'gpt3':
        generations_iter = gpt3(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file)
    elif model_type == 'gpt2-greedy':
        print("Using n=1 for greedy generation (sampling does not apply)")
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=1,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file,
                                sample=False)
    elif model_type == 'gpt2-naughty-list':
        # Load pre-tokenized naughty words
        # FIXME: output dir must already exist with this file
        with open(output_dir / 'gpt2_naughty_token_ids.pkl', 'rb') as f:
            naughty_list_ids = pickle.load(f)
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file,
                                bad_words_ids=naughty_list_ids)
    elif model_type == 'gpt2-affect':
        generations_iter = gpt2_affect(prompts=prompts,
                                       max_len=max_tokens,
                                       num_samples=n,
                                       batch_size=batch_size,
                                       model_name_or_path=model,
                                       out_file=generations_file,
                                       # Affect
                                       target_class=0,
                                       num_classes=2,
                                       beta=1)
    elif model_type == 'gpt2-ctrl':
        generations_iter = gpt2_ctrl(prompts=prompts,
                                     max_len=max_tokens,
                                     num_samples=n,
                                     batch_size=batch_size,
                                     model_name_or_path=model,
                                     out_file=generations_file,
                                     # GPT2-CTRL
                                     prompt_ctrl_code='<|nontoxic|>')
    elif model_type == 'openai-gpt':
        generations_iter = openai_gpt(prompts=prompts,
                                      max_len=max_tokens,
                                      num_samples=n,
                                      model_name_or_path=model,
                                      out_file=generations_file)
    elif model_type == 'ctrl':
        assert model == 'ctrl'
        generations_iter = ctrl(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                model_name_or_path=model,
                                out_file=generations_file,
                                # CTRL
                                ctrl_code='Links',
                                temperature=1.0,
                                repetition_penalty=1.2)
    elif model_type == 'pplm':
        generations_iter = pplm(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                class_label=0,
                                num_iterations=10,
                                model_name_or_path='toxicity',
                                out_file=generations_file)
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate and collate perspective scores
    generations = []


    for i, gen in enumerate(generations_iter):
        generations.append(gen)
        perspective(f'generation-{i}', gen)


    torch.cuda.empty_cache()
    perspective.stop()
    print('Finished generation and perspective scoring!')

    print('Collating output files')
    collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()
