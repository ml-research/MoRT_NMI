python scripts/run_prompts_experiment.py "./resultsMCM/final/eos-gpt2mcm-k50-keepmin5-t0.0-samples10000" --model gpt2 --model-type gpt2mcm --use-eos --n 10000 --mcm-t 0.0 --top-k 50 --mcm-keepmin 5
python scripts/run_prompts_experiment.py "./resultsMCM/final/eos-gpt2mcm-k50-keepmin5-t0.5-samples10000" --model gpt2 --model-type gpt2mcm --use-eos --n 10000 --mcm-t 0.5 --top-k 50 --mcm-keepmin 5
python scripts/run_prompts_experiment.py "./resultsMCM/final/eos-gpt2mcm-k50-keepmin5-tneg0.5-samples10000" --model gpt2 --model-type gpt2mcm --use-eos --n 10000 --mcm-t -0.5 --top-k 50 --mcm-keepmin 5