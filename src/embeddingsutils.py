import os
import gc
import psutil
import numpy as np
import streamlit as st
from gensim.models import FastText, Word2Vec, KeyedVectors
from gensim.models.fasttext import load_facebook_model

def check_and_free_memory(threshold_percent=80.0):
    mem = psutil.virtual_memory()
    if mem.percent > threshold_percent:
        st.cache_resource.clear()
        gc.collect()

def get_model_files(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    valid_extensions = ('.model', '.vec', '.txt', '.bin', '.kv', '.glove')
    return [f for f in os.listdir(directory) if f.endswith(valid_extensions)]

def get_wv(model):
    return getattr(model, 'wv', model)

def get_sentence_vector(text, wv):
    words = text.lower().split()
    vectors = []
    for w in words:
        try:
            vectors.append(wv[w])
        except KeyError:
            continue
    if len(vectors) == 0:
        return np.zeros(wv.vector_size)
    return np.mean(vectors, axis=0)

@st.cache_resource
def load_embedding_model(filepath):
    check_and_free_memory(80.0)
    loaders = [
            lambda p: KeyedVectors.load(p),
            lambda p: FastText.load(p),
            lambda p: Word2Vec.load(p),
            lambda p: load_facebook_model(p),
            lambda p: KeyedVectors.load_word2vec_format(p, binary=False),
            lambda p: KeyedVectors.load_word2vec_format(p, binary=True),
            lambda p: KeyedVectors.load_word2vec_format(p, binary=False, no_header=True),
        ]

    for loader in loaders:
        try:
            return loader(filepath)
        except Exception:
            continue
    return None

def get_similar_words(wv, query_word, metric="cosine", topn=15):
    if metric == "cosine":
        return wv.most_similar(query_word, topn=topn)

    query_vec = wv[query_word]
    all_vectors = wv.vectors
    index_to_key = wv.index_to_key

    if metric == "dot":
        scores = np.dot(all_vectors, query_vec)
        best_indices = np.argsort(scores)[::-1][:topn+1]
    elif metric == "euclidean":
        diff = all_vectors - query_vec
        distances = np.linalg.norm(diff, axis=1)
        best_indices = np.argsort(distances)[:topn+1]
    elif metric == "manhattan":
        diff = all_vectors - query_vec
        distances = np.sum(np.abs(diff), axis=1)
        best_indices = np.argsort(distances)[:topn+1]
    elif metric == "chebyshev":
        diff = all_vectors - query_vec
        distances = np.max(np.abs(diff), axis=1)
        best_indices = np.argsort(distances)[:topn+1]
    else:
        raise ValueError("Unknown metric")
    results = []
    for idx in best_indices:
        word = index_to_key[idx]
        if word != query_word:
            if metric == "dot":
                val = float(scores[idx])
            else:
                val = float(distances[idx])
            results.append((word, val))
        if len(results) == topn:
            break

    return results
