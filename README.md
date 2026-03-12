# Patronizing and Condescending Language Detection (Multi-label Classification)

This repository contains a multi-label NLP text classification project for detecting **Patronizing and Condescending Language (PCL)**.

PCL is language that adopts a *superior* or *pitying* tone toward individuals or communities, often while the author intends to be helpful (e.g., raising awareness or funds). Even when unintentional, this framing can reinforce stereotypes and normalize discrimination by making it feel “routine” or socially acceptable.

## Dataset

The dataset comes from the **Don’t Patronize Me!** project and can be obtained from the original repository:
- https://github.com/Perez-AlmendrosC/dontpatronizeme

### File
- `dontpatronizeme_categories.tsv`

### Format
The dataset contains **993 paragraphs** annotated for **multi-label classification**.

Each example has a **7-dimensional one-hot label vector**. The *i-th* entry is `1` if the corresponding category is present in the paragraph; otherwise it is `0`. A paragraph may contain one or more PCL spans and can be assigned **multiple labels**.

### Label set (7 categories)
1. Unbalanced Power Relations
2. Shallow Solution
3. Presupposition
4. Authority Voice
5. Metaphor
6. Compassion
7. The Poorer, the Merrier

## Project tasks

1. **Feature extraction**
   - TF–IDF
   - Pretrained **Google Word2Vec** embeddings (Mikolov et al., 2013), **200-dimensional**

2. **Multi-label classification (Statistical NLP)**

3. **Performance analysis**

## Requirements

- Python **3.10+**
- scikit-learn **0.24.1**
- scikit-multilearn
- gensim **4.1.2**
- NLTK

> Tip: It’s recommended to create a virtual environment before installing dependencies.

## Usage

Training is done via command line. In the examples below, replace `classifier.py` with the training script you want to run.

### Command template

```bash
python3 <filename>.py [OPTIONS]
```

### Arguments

| Parameter | Description |
|---|---|
| `--load` | Path to the raw `.tsv` or `.csv` file. Example: `--load /path/to/dontpatronizeme_categories.tsv` |
| `--name`, `-n` | Classifier type. Supported: `classifier_chains`, `LP` (Label Powerset), `BR` (Binary Relevance). Example: `--name BR` |
| `--input_type` | Feature extraction method: `tfidf` or `gword2vec`. Example: `--input_type tfidf` |
| `--preprocess` | Preprocessing option: `l` (lemmatize), `s` (stemming), `n` (none). Example: `--preprocess n` |
| `--stopwords` | If set, removes stopwords. (Flag; no value needed.) |
| `--features` | Number of features to use. Example: `--features 200` |
| `--save` | If set, saves the trained model. (Flag; no value needed.) |

### Example

Binary Relevance with Google Word2Vec, lemmatization, stopword removal, 200 features, and saving the model:

```bash
python3 classifier.py \
  --load /x/y/z/dontpatronizeme_categories.tsv \
  --name BR \
  --input_type gword2vec \
  --preprocess l \
  --stopwords \
  --features 200 \
  --save
```

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{perez-almendros2020dontpatronizeme,
  title={Don’t Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities},
  author={Perez-Almendros, Carla and Espinosa-Anke, Luis and Schockaert, Steven},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={5891--5902},
  year={2020}
}
```