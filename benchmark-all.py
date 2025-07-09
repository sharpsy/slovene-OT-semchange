import collections
import pickle

import numpy as np
import ot
import pandas as pd
import scipy
import sklearn
import tqdm


scores_df = pd.read_csv("data/semantic_shift_scores.tsv", sep="\t")

with open("out/T1-vecs.pkl", "rb") as T1_file:
    word_tokens_T1 = pickle.load(T1_file)

with open("out/T2-vecs.pkl", "rb") as T2_file:
    word_tokens_T2 = pickle.load(T2_file)


def PRT_score(t1_vecs, t2_vecs):
    t1_vecs = np.stack(t1_vecs).mean(axis=0)
    t2_vecs = np.stack(t2_vecs).mean(axis=0)
    dist = scipy.spatial.distance.cosine(t1_vecs, t2_vecs)
    return 1 / (1 - dist).item()


def APD_score(t1_vecs, t2_vecs):
    cdist = scipy.spatial.distance.cdist(t1_vecs, t2_vecs, "cosine")
    return cdist.mean()


def APDM_score(t1_vecs, t2_vecs):
    V1 = np.stack(t1_vecs)
    V1 = V1.T / np.linalg.norm(V1, axis=1)
    v1 = V1.mean(axis=1)

    V2 = np.stack(t2_vecs)
    V2 = V2.T / np.linalg.norm(V2, axis=1)
    v2 = V2.mean(axis=1)

    return 1 - (np.linalg.norm(v1) * np.linalg.norm(v2))


def APDD_score(t1_vecs, t2_vecs):
    # print("apd")
    V1 = np.stack(t1_vecs)
    V1 = V1.T / np.linalg.norm(V1, axis=1)
    v1 = V1.mean(axis=1)

    V2 = np.stack(t2_vecs)
    V2 = V2.T / np.linalg.norm(V2, axis=1)
    v2 = V2.mean(axis=1)

    return scipy.spatial.distance.cosine(v1, v2)


def OT_score(t1_vecs, t2_vecs):
    cdist = scipy.spatial.distance.cdist(t1_vecs, t2_vecs, "cosine")
    return ot.lp.emd2([], [], cdist)


def OT_regularized_score(t1_vecs, t2_vecs):
    cdist = scipy.spatial.distance.cdist(t1_vecs, t2_vecs, "cosine")
    return ot.bregman.sinkhorn2([], [], reg=0.2, M=cdist)


def JSD_score(t1_vecs, t2_vecs):
    kmeans = sklearn.cluster.KMeans(5, n_init=5)
    preds = kmeans.fit_predict(t1_vecs + t2_vecs)

    c1 = collections.Counter()
    for i in set(preds):
        c1[i] = 0
    c2 = c1.copy()

    c1.update(preds[: len(t1_vecs)])
    c2.update(preds[len(t2_vecs) :])
    return scipy.spatial.distance.jensenshannon(list(c1.values()), list(c2.values()))


all_words = set(word_tokens_T1)


for desc, score_func in [
    ("APD", APD_score),
    ("OT", OT_score),
    ("OT-reg-02", OT_regularized_score),
    ("JSD", JSD_score),
    ("PRT", PRT_score),
]:
    col_layers = []
    col_words = []
    col_res = []

    for layers in tqdm.tqdm(
        [[k] for k in range(13)] + [[9, 10, 11, 12], [9, 10, 11], [10, 11, 12]]
    ):
        the_results = {}

        for word in tqdm.tqdm(all_words, leave=False):
            t1_vecs = []
            t2_vecs = []

            for word_toks in word_tokens_T1[word]:
                t1_vecs.append(word_toks[layers].mean(axis=0))

            for word_toks in word_tokens_T2[word]:
                t2_vecs.append(word_toks[layers].mean(axis=0))
            the_results[word] = score_func(t1_vecs, t2_vecs)

            col_layers.append(str(layers))
            col_words.append(word)
            col_res.append(the_results[word])

        calculated_scores = [the_results[r.word] for r in scores_df.itertuples()]
        R = scipy.stats.spearmanr(calculated_scores, 4 - scores_df.score.values)
        print(layers, f"{desc} ---", R.correlation)

    pd.DataFrame(
        {
            "layers": col_layers,
            "words": col_words,
            "apd_score": col_res,
        }
    ).to_csv(f"out/results-{desc}.csv", index=False)
