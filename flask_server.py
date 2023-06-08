#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/11/02 00:06:44
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''


import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import yaml
import logging
import data_input
logging.basicConfig(level=logging.INFO)
import data_input
from config import Config
from model.siamese_network import SiamenseRNN, SiamenseBert
from data_input import Vocabulary, get_test
from util import write_file
from flask import Flask
import json

import  re

from faiss_index_recall import *

from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/query_recall2/<q1>', methods=['get'])
def query_recall2(q1):

    test_arr, query_arr = data_input.get_test_bert_singletext(q1, vocab)

    test_emb = model.predict_embedding(test_arr)

    test_emb = np.array(test_emb)

    texts = get_knn(index, test_emb, id_texts, 15)

    test_arr, query_arr = data_input.get_query_recall_bert(q1, texts, vocab)

    test_label, test_prob = model.predict_nomerge(test_arr)

    pro_txt = list(zip(test_prob, texts))
    pro_txt=sorted(pro_txt, key=lambda x: x[0], reverse=True)
    print(q1)
    print(pro_txt)
    rets=[]
    for p_t in pro_txt:
        pro=p_t[0]
        text=p_t[1]
        rets.append(f"概率得分：\t{pro}\t,检索内容：\t{text}")
    rets="\n".join(rets)

    return json.dumps({"info":"返回成功","data":rets}, ensure_ascii=False)


if __name__ == '__main__':
    cfg_path = "./configs/config_bert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    model = SiamenseBert(cfg)
    model.restore_session(cfg["checkpoint_dir"])

    path = "./data/query_emb_all.txt.siamese.bert.embedding"
    id_texts, query_embedding=create_embedding_fromfile(path)
    index=create_faiss_index(query_embedding)
    app.run("0.0.0.0",80)

    
    
    # http://192.168.1.23:8080//query_recall/苹果手机
    #http://127.0.0.1/sentence_insert_into_database/你好吗
    #http://127.0.0.1/sentence_insert_into_database/你好呀
    #http://127.0.0.1/query_sim_sentence/近期上映的电影/2
    
    
    # sentence_insert_into_database("1","近期上映的电影")
    # sentence_insert_into_database("2","近期上映的电影有哪些")
    # sentence_insert_into_database("3","杭州哪里好玩")
    # sentence_insert_into_database("4","杭州哪里好玩点")
    # sentence_insert_into_database("5","这是什么乌龟值钱吗")
    # sentence_insert_into_database("6","什么东西越热爬得越高")
    # sentence_insert_into_database("7","长的清新是什么意思")
    # sentence_insert_into_database("8","淘宝上怎么用信用卡分期付款")
    # app.run("0.0.0.0",80)
    
    
    # query_sim_sentence_bm25_recall("怎么样学习好物理","1,2",5)

    # query_bm25_recall("怎么样学习好物理","1,2",5)
    
