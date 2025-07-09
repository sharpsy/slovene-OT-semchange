import collections
import pickle

import pandas as pd
import spacy
import tqdm
from lemmagen3 import Lemmatizer


# Load data
df = pd.read_csv("word_usage_annotations_1997_2018.tsv", sep="\t")
target_words = set(df.word.unique().tolist())

_GIGAFIDA_PATH = "slovenian_semantic_shift_dataset/gigafida_to_1997_vs_2018.tsv"

giga_df = pd.read_csv(_GIGAFIDA_PATH, sep="\t")

T1 = giga_df[giga_df.date < 2018]
T2 = giga_df[giga_df.date == 2018]


# Initialize lemmatizer, spacy pipeline
T1_sents_by_words = collections.defaultdict(list)
T2_sents_by_words = collections.defaultdict(list)


lem_sl = Lemmatizer("sl")

nlp = spacy.load(
    "sl_core_news_sm",
    exclude=["lemmatizer", "senter", "attribute_ruler", "ner"],
)
nlp.max_length = 10000000


# Process text, cut into sentences, match lemma and store to the file for later rpocessing

processed_T1_gen = nlp.pipe(T1.text.tolist(), batch_size=30, n_process=14)

for doc in tqdm.tqdm(T1.itertuples(), total=len(T1)):
    processed_text = next(processed_T1_gen)
    for s in processed_text.sents:
        sent_lemmas = set(map(lem_sl.lemmatize, s.text.lower().split()))
        for w_match in sent_lemmas & target_words:
            T1_sents_by_words[w_match].append(s.text)


with open("T1.pkl", "wb") as T1_file:
    pickle.dump(T1_sents_by_words, T1_file)

processed_T2_gen = nlp.pipe(T2.text.tolist(), batch_size=4, n_process=8)

for doc in tqdm.tqdm(T2.itertuples(), total=len(T2)):
    processed_text = next(processed_T2_gen)
    for s in processed_text.sents:
        sent_lemmas = set(map(lem_sl.lemmatize, s.text.lower().split()))
        for match in sent_lemmas & target_words:
            T2_sents_by_words[match].append(s.text)

with open("T2.pkl", "wb") as T2_file:
    pickle.dump(T2_sents_by_words, T2_file)
