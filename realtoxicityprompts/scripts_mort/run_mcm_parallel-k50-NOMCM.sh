python scripts/run_prompts_experiment.py "./resultsMCM/prompted/gpt2mcm-k50-NOMCM" --model gpt2 --model-type gpt2mcm --mcm-t 0.5 --top-k 50 --mcm-keepmin 5 --n 10 --no-usemcm --dataset-file ./data/realtoxicityprompts/extracted/realtoxicityprompts-data/prompts.jsonl --perspective-rate-limit 25