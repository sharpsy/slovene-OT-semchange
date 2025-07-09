# Tracking Semantic Change in Slovene

This repository contains the code and resources for the paper **"Tracking Semantic Change in Slovene: A Novel Dataset and Optimal Transport-Based Distance"**, available on [arXiv](https://arxiv.org/abs/2402.16596v2).

## Abstract

This research addresses the detection of semantic change in Slovene, a less-resourced Slavic language spoken by two million people. Understanding how word meanings evolve offers insight into societal and cultural shifts. We introduce the first Slovene dataset for semantic change evaluation, featuring aggregated change scores for 104 target words derived from over 3,000 manually annotated sentence pairs.

We critically analyze the widely used *Average Pairwise Distance (APD)* metric and identify its limitations. As a solution, we propose a novel metric based on **regularized optimal transport**, which provides a more robust and theoretically grounded framework for quantifying semantic change. Our comprehensive evaluation of existing semantic change detection methods demonstrates that the proposed method matches or outperforms current baselines.

## Dataset

The **"Semantic change detection datasets for Slovenian 1.0"**, available via [CLARIN.SI](http://hdl.handle.net/11356/1651), is the first dataset for evaluating semantic change detection in Slovene. It includes:

* **Source corpus** (`gigafida_to_1997_vs_2018.tsv`): Texts from the Gigafida 2.0 corpus, split into two time periods (1997 or earlier, and 2018), useful for training or embedding extraction.
* **Annotated sentence pairs** (`word_usage_annotations_1997_2018.tsv`): 3,150 sentence pairs (30 per time period) for 105 target words, manually rated by 3 annotators on a 1–4 semantic similarity scale.
* **Gold standard scores** (`semantic_shift_scores.tsv`): Aggregated semantic change scores for 104 words, used as ground truth for system evaluation.
* **Annotation guidelines** (`RSDO_semanticni-premiki_navodila_v0.pdf`): Instructions for human annotation (in Slovene).

The dataset provides a reliable benchmark for tracking and evaluating semantic change in a less-resourced language. Detailed information on data collection, annotation, and validation is provided in the paper.

## Repository Structure and Code Overview

This repository includes scripts for preprocessing, vector extraction, and evaluation:

* **`gigafida-extract.py`**
  Loads the GigaFida corpus, splits it into sentences, and extracts all sentences containing a given target word from two different time periods. It outputs two files mapping each target word to a list of sentences.

* **`infer-word-vectors.py`**
  Uses the **SloBERTa** language model to obtain contextual embeddings of target words from the extracted sentences. The vectors are stored in files as mappings from words to their vector representations.

* **`benchmark-all.py`**
  Compares various semantic change measures using the extracted vectors. It evaluates each method by computing the **Spearman's rank correlation** between predicted change scores and ground-truth rankings.

## Getting Started

### 1. Download the Dataset

Download the dataset from the [CLARIN.SI repository](http://hdl.handle.net/11356/1651) and place the extracted contents in the root directory of this repository.

### 2. Set Up the Environment

Install the required Python packages:

```bash
pip install numpy scipy pandas pot torch transformers tqdm spacy lemmagen3 rapidfuzz
```

Ensure you are using some type of Python virtual environment so you do not overwrite or bloat your existing python libraries. 

### 3. Run the Pipeline

Some file paths might need corrections, but that should be a minor change.

1. **Extract Sentences**

   ```bash
   python gigafida-extract.py
   ```

2. **Infer Word Vectors**

   ```bash
   python infer-word-vectors.py
   ```

3. **Evaluate Semantic Change Measures**

   ```bash
   python benchmark-all.py
   ```

The output will include evaluation scores and rankings for different semantic change detection methods.

## Methodology Highlights

Our proposed approach applies **regularized optimal transport** to compute semantic distances between word usage distributions across two time periods. This framework addresses limitations of traditional methods like average pairwise distance by accounting for word usage distribution alignment in a principled way.

## Citation

If you use this code or dataset in your work, please cite the following:

**Paper:**

```bibtex
@misc{pranjić2025trackingsemanticchangeslovene,
      title={Tracking Semantic Change in Slovene: A Novel Dataset and Optimal Transport-Based Distance}, 
      author={Marko Pranjić and Kaja Dobrovoljc and Senja Pollak and Matej Martinc},
      year={2025},
      eprint={2402.16596},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.16596v2}
}
```

**Dataset:**

```bibtex
@misc{11356/1651,
  title = {Semantic change detection datasets for {S}lovenian 1.0},
  author = {Martinc, Matej and Dobrovoljc, Kaja and Pollak, Senja},
  url = {http://hdl.handle.net/11356/1651},
  note = {Slovenian language resource repository {CLARIN}.{SI}},
  copyright = {The {MIT} License ({MIT})},
  issn = {2820-4042},
  year = {2022}
}
```
