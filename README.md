# Semantic Change Detection for Slovene Language

Welcome to the repository containing the code to reproduce results for the paper "Semantic change detection for Slovene language: a novel dataset and an approach based on optimal transport". This repository provides the necessary resources to reproduce the findings of our study, as outlined in our paper available at [arXiv](https://arxiv.org/abs/2402.16596).

## Abstract

In this paper, we focus on the detection of semantic changes in Slovene, a less resourced Slavic language with two million speakers. Detecting and tracking semantic changes provides insights into the evolution of the language caused by changes in society and culture. Recently, several systems have been proposed to aid in this study, but all depend on manually annotated gold standard datasets for evaluation. In this paper, we present the first Slovene dataset for evaluating semantic change detection systems, which contains aggregated semantic change scores for 104 target words obtained from more than 3000 manually annotated sentence pairs. We evaluate several existing semantic change detection methods on this dataset and also propose a novel approach based on optimal transport that improves on the existing state-of-the-art systems with an error reduction rate of 22.8%. 

## Dataset

The "Semantic change detection datasets for Slovenian 1.0" dataset, is publicly available at the [CLARIN.SI repository](http://hdl.handle.net/11356/1651). It represents the first Slovene dataset dedicated to the evaluation of semantic change detection. The compilation and annotative methodology behind the dataset are comprehensively detailed in our paper.

## Reproducing the Results

To reproduce the results presented in our paper, please follow the steps outlined below:

1) Download the Dataset: The "Semantic change detection datasets for Slovenian 1.0" is required to be present locally to run the experiments. Visit  [CLARIN.SI repository](http://hdl.handle.net/11356/1651) to download the dataset. After downloading, unzip and extract the files alongside the notebook from this repo.
2) Set Up Your Environment and Install Dependencies: Make sure you have a Jupyter Python environment with the packages used in the notebook (numpy, scipy, pandas, pot, torch, transformers, tqdm, thefuzz).
3) Run Jupyter Notebook and Execute the Notebook

## Approach

Our approach applies optimal transport to the domain of semantic change detection. Our method systematically measures the discrepancies in word usage distributions over different time slices, thus capturing the dynamics of semantic evolution.

## How to Cite

To reference our paper in your academic work, please use the following citation:

```bibtex
@misc{pranjic2024semantic,
      title={Semantic change detection for {S}lovene language: a novel dataset and an approach based on optimal transport}, 
      author={Marko Pranjić and Kaja Dobrovoljc and Senja Pollak and Matej Martinc},
      year={2024},
      eprint={2402.16596},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

When utilizing the dataset, kindly include the following citation:

```bibtex
 @misc{11356/1651,
 title = {Semantic change detection datasets for {S}lovenian 1.0},
 author = {Martinc, Matej and Dobrovoljc, Kaja and Pollak, Senja},
 url = {http://hdl.handle.net/11356/1651},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {The {MIT} License ({MIT})},
 issn = {2820-4042},
 year = {2022} }
```

Thank you for your interest in our work and for supporting reproducible research within the computational linguistics community. We look forward to seeing how our dataset and approach to semantic change detection in the Slovene language are utilized and expanded upon in future studies.
