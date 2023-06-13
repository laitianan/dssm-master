# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:15:20 2023

@author: 98608
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import csv
import faiss



def create_embedding_fromfile(path):
    reader = csv.reader(open(path, encoding="utf-8"), delimiter='\t')
    query_embedding = []
    id_texts = dict()
    for i, line in tqdm(enumerate(reader)):
        title = line[0]
        emb = list(eval(line[1]))
        query_embedding.append(emb)
        id_texts[i] = title
        # break
    query_embedding = np.array(query_embedding)
    query_embedding = query_embedding.astype('float32')
    return id_texts, query_embedding

def create_faiss_index(reference_embeddings):

    d = reference_embeddings.shape[1]

    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:  # GPU
        print("[INFO]faiss gpu: {}".format(faiss.get_num_gpus()))
        # index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)  #
    return index

def get_knn(index,query_embedding, id_texts,k):
    query_embedding=query_embedding.astype('float32')
    _, indices = index.search(query_embedding, k + 1)
    texts = []
    for i in indices[0]:
        text=id_texts[i].split("####")[-1]
        texts.append(text)
    return texts


if __name__ == '__main__':
    path = "./data/query_emb_all.txt.siamese.bert.embedding"
    id_texts, query_embedding=create_embedding_fromfile(path)
    index=create_faiss_index(query_embedding)
    texts=get_knn(index, query_embedding[:1], id_texts,5)
    print(texts)