
# Patronizing and Condescending Language Detection (Multilabel classification)
In this project we deal with the class of NLP classification problems called Patronizing and Conde- scending Language (PCL) Detection. Before that i would like to talk in brief, about the motivation behind these sets of problems. When one engages in PCL ,the language use shows a superior attitude towards others or depicts them in a compassionate way. This effect is not always conscious and the intention of the author is often to help the person or group they refer to (e.g. by raising awareness or funds, or moving the audience to action). However, these superior attitudes and a discourse of pity can routinize discrimination and make it less visible

## Data description
All data is available upon request at [this](https://github.com/Perez-AlmendrosC/dontpatronizeme) repository.

The file ’dontpatronizeme categories.tsv’ is the PCL multilabel classification dataset. It contains 993 paragraphs again annotated with labels in a one hot vector array form. There are seven categories namely :-(i) Unbalanced Power Relations (ii)Shallow Solution (iii)Presupposition (iv) Authority Voice (v)Metaphor (vi) Compassion (vii)The poorer the merrier. The i’th entry in the array is 1 if that category is detected in a text span, otherwise it’s 0.Each paragraph might contain one or more text spans with PCL, which may be assigned to the same or to different categories.


## Tasks at hand :-

(i) Feature Extraction - We employ TF-IDF and the pretrained Google Word2Vec (Mikolov et. Al 2013) with a 200 dimensional vector.  

(ii)Multilabel Classification using Statistical NLP. 

(iii) Performance Analysis




## 🛠 Skills
Python 3


## Citations
@inproceedings{perez-almendros2020dontpatronizeme,
  title={Don’t Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities},
  author={Perez-Almendros, Carla and Espinosa-Anke, Luis and Schockaert, Steven},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={5891--5902},
  year={2020}
}
## How to Use
README FILE
 
Patronized Condescending Language

Read this file before running the code inorder to understand commands required to run the code.

##System dependencies

—python==3.10 or above

—Scikit-learn==0.24.1

—Scikit-multilearn

—gensim==4.1.2

-NLTK

## Here we provide commands needed to train the model, just append whatever processing you want to do to the command (python3 filename.py ) on terminal:

| Parameters       |         Description |
| -------------    | --------------------|
 
| ``—-load``          | path to raw '.tsv' file or '.csv' file  e.g --load /DATA1/NLP/XXXX/YYYY/dontpatronizeme_categories.tsv (give the path to csv)|   
|
| ``--name``, ``-n``  | classifier to use classifierchains('classifier_chains' ), Label Powerset('LP'), Binary Relevance ('BR') ,e.g -name BR  |

| ``—input_type``     | what type of feature extraction do you want (tf-idf or gword2vec)  ,e.g --input_type tfidf (if you want tfidf) |

| ``—preprocess``     | preprocessing you want lemmatize('l'), stemming('s') and no-preprocessing ('n') e.g --preprocess n (if you don't want any prepprocessing)|

| ``—stop words``     | if you want to remove stopwords or not ,e.g --stopwords (if you want to remove stopwords) |

| ``—features``       | number of features you want ,e.g --features 200 |

| ``--save``          | to save the classifier model or not ,e.g --save (if you want to save the model)|



## example running files

If you want to run Binary Relevance classifier , feature extraction technique is gword2vec, save the model, preprocessing is lemmatisation , number of features are 200 and you want to remove stop words and path to csv is /x/y/z then command will be as follows, if classifier.py is the file:

-->   python3 classifier.py --load /x/y/z --name BR --input_type gword2vec --preprocess l --stopwords --features 200 --save
Footer
