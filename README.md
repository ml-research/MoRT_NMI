# Pre-Trained Language Models Mirror Human-like Moral Norms
Code repository and Supplemental Material for the "Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do" 
publication.

## 1. Structure
The code is structured in:

* /MoRT: the moral choice machine and moral direction source code
* /LAMA: forked repo of https://github.com/facebookresearch/LAMA
* /realtoxicityprompts forked repo of https://github.com/allenai/real-toxicity-prompts
* /Supplemental_Material additional supplement (see below)

The Supplemental Material is divided in:

* /MoralScore: tables with results to reproduce the manuscript figures
* /TextGeneration: results of text generation. Read the contrained README.md for download links.
* /UserStudy: material of conducted user studies

## 2. Dependencies
    see MoRT/requirements.txt
    see LAMA/requirements.txt
    see realtoxicityprompts/environment.yml


## 3. Before you start 
* create an virtual enviroment and install the requirements
* download requiered data for LAMA (see README https://github.com/facebookresearch/LAMA) by executing LAMA/download_models.sh
* download testbed data from https://open.quiltdata.com/b/ai2-datasets/tree/realtoxicityprompts/ and place it in /real-toxicity-prompts/data/
* download text generation with the moral direction approach from https://hessenbox.tu-darmstadt.de/public?folderID=MjR2QVhvQmc0blFpdWd1YjViNHpz

## 4. Reproducing Results
Scripts and pipeline to reproduce results. Steps with (optional) are only required to reproduce data which is already contained in this repository.
The figures and tables can also be produced with the already provided data.

### LAMA
Create conda or virtual environment and install requirements (also see instruction of https://github.com/facebookresearch/LAMA)

    cd LAMA
    python lama/eval_generation_moral.py --lm "bert"

### PCA MoralDirection (optional)
Create and save transformation
Create conda or virtual environment and install requirements

    cd MoRT
    python mort/pca.py  --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 --bert_model_name bert-large-nli-mean-tokens
If this file should be used for the next steps instead of the provided one, place it in MoRT/data/subspace_proj/bert-large-nli-mean-tokens/projection_model.p

### Compute Moral Scores (optional)
    cd MoRT
    export PYTHONPATH=.

Universal Sentence Encoder:

    python mort/plot_corr/compute_mcm_cossim_scores.py --data userstudy --model use_hub
    python mort/plot_corr/compute_mcm_cossim_scores.py --data pos50 --model use_hub
    python mort/plot_corr/compute_mcm_cossim_scores.py --data neg50 --model use_hub
GloVe and BERT:

    python mort/plot_corr/computeBERTScorePosNeg.py
    python mort/plot_corr/computeGloveScorePosNeg.py
    python mort/plot_corr/computeBERTAndGloVeScoreOfUserStudyActions.py

### Correlation between human and PLM's moral scores

    python mort/plot_corr/plt_UserStudyCorr.py

By default the correlation is calculated with the regional study. 
To apply the global AMT study replace 'regional' with 'globalAMT' in line 181.

    

### Realtoxicityprompts text generation (optional)
Run scripts contained in realtoxicityprompts/scripts_mort 

### Realtoxicityprompts Evaluation
Before you run the notebooks change the path to this repository.

- Run notebook for figures and result table: realtoxicityprompts/notebooks_mcm/realtoxicityprompts_results_moral.ipynb .

- Run notebook for inspecting selected examples: realtoxicityprompts/notebooks_mcm/example_generaions.ipynb .

## 5. Try MD
You can compute moral score of queries by running following script and typing your query, e.g. "Should I kill", "Should I kill time".

    python mort/runMCM.py --model orig
Type stop to exit.

## 6. Citing & Authors

If you find this repository helpful, feel free to cite our publication: TODO
