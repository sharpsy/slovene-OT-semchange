import collections
import pickle
import re

import numpy as np
import torch
import tqdm
import transformers
from lemmagen3 import Lemmatizer
from rapidfuzz import fuzz, process


lem_sl = Lemmatizer("sl")


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "EMBEDDIA/sloberta", use_fast=True
)
model = transformers.AutoModelForMaskedLM.from_pretrained(
    "EMBEDDIA/sloberta", output_hidden_states=True, device_map="cuda"
)


def preprocessor(str):
    return lem_sl.lemmatize(str).lower()


with open("data/T1.pkl", "rb") as T1_file:
    T1_sents_by_words = pickle.load(T1_file)

with open("data/T2.pkl", "rb") as T2_file:
    T2_sents_by_words = pickle.load(T2_file)


def _get_word_tokens(model, tokenizer, sentence, word):
    _word, score, _pos = process.extractOne(
        word,
        re.split(r"\W+", sentence),
        scorer=fuzz.QRatio,
        processor=preprocessor,
    )
    word_start_ix = sentence.find(_word)
    word_end_ix = word_start_ix + len(_word)
    assert word_start_ix >= 0
    assert score > 50, (score, word, _word, sentence)
    # tokenize one by one so we avoid padding > 1/2 zeros (bisect won't work)
    tokenized = tokenizer(
        sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True
    )

    offsets = tokenized["offset_mapping"][0]
    start_off = offsets[:, 0].contiguous()
    end_off = offsets[:, 1].contiguous()

    if word_end_ix > end_off[-2].item():
        return None

    # [1:-1] -- ignore initial/final token
    start_tok = torch.searchsorted(start_off[1:-1], word_start_ix, side="right")
    # start_tok is advanced by 1, but we are indexing to include it
    # (ie. start from start_tok-1) so it cancels itself out

    # [:-1] ignore initial/final token
    end_tok = torch.searchsorted(end_off[:-1], word_end_ix) + 1

    with torch.no_grad():
        tokenized.pop("offset_mapping")
        hs = model(**tokenized.to("cuda"))
        tokens_by_layers = [
            hs_hidden[0, start_tok:end_tok].mean(axis=0).cpu()
            for hs_hidden in hs.hidden_states
        ]
        tokens_by_layers = np.stack(tokens_by_layers)
    return tokens_by_layers / np.linalg.norm(tokens_by_layers, axis=0)


word_tokens_T1 = collections.defaultdict(list)
word_tokens_T2 = collections.defaultdict(list)

words = set(T1_sents_by_words)


for word in tqdm.tqdm(words):
    for sentence in tqdm.tqdm(T1_sents_by_words[word], leave=False):
        tokens_by_layers = _get_word_tokens(model, tokenizer, sentence, word)
        if tokens_by_layers is not None:
            word_tokens_T1[word].append(tokens_by_layers)

    for sentence in tqdm.tqdm(T2_sents_by_words[word], leave=False):
        tokens_by_layers = _get_word_tokens(model, tokenizer, sentence, word)
        if tokens_by_layers is not None:
            word_tokens_T2[word].append(tokens_by_layers)


with open("out/T1-vecs.pkl", "wb") as T1_file:
    pickle.dump(word_tokens_T1, T1_file)

with open("out/T2-vecs.pkl", "wb") as T2_file:
    pickle.dump(word_tokens_T2, T2_file)
