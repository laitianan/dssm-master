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
from LAC import LAC
import  re
import sys, os, pathlib

import sys, os, pathlib
root = pathlib.Path(os.path.abspath(__file__)).parent
bm25_root = os.path.join(root, 'bm25')
sql_root = os.path.join(root, 'sqlclient')

es_root = os.path.join(root, 'esclient')

sys.path.extend([bm25_root, sql_root,es_root])

from sqlclient.sqlhelper import SqlClient
from bm25.bm25_recall import Bm25Recall
from esclient.elasticsearch_helper import ESClient

app = Flask(__name__)

def lac_cut(s):
    global lac
    s=re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]"," ",s)
    return ' '.join(lac.run(s))

def cut(s):
    s = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", s)
    return ' '.join(list(s))

@app.route('/query_sim/<cut>/<q1>/<q2>', methods=['GET'])
def query_sim(cut,q1,q2):
    print(cut,q1,q2)

    test_arr2, query_arr2  =data_input.get_query_bert(q1,q2, vocab)
    label, prob = model.predict_nomerge(test_arr2)
    prob_semantic=prob[0]
    if cut=="lac":
        prob_not_semantic=jaccard_similarity(q1,q2, True)
    else:
        prob_not_semantic = jaccard_similarity(q1, q2, False)
    retjson={"query1":q1,"query2":q2,"prob_semantic":float(prob_semantic),"prob_not_semantic":float(prob_not_semantic)}
    return json.dumps(retjson, ensure_ascii=False)

def insert_database(qids,q1s,embs):
    global database,sentences_dict
    database=embs
    for i in range(len(qids)):
        sentences_dict[i]=qids[i],q1s[i]
    return True
        



# @app.route('/sentence_insert_into_database/<qid>/<q1>', methods=['GET'])
def sentence_insert_into_database(qids,q1s):
    global  vocab
    # test_arr, query_arr = data_input.get_test_bert_singlequery(q1s, vocab)
    # item_embs=model.predict_embedding(test_arr)
    
    # item_embs=np.reshape(item_embs,[-1,768])
    
    item_embs=None
    isinsert=insert_database(qids,q1s,item_embs)
    return isinsert


# @app.route('/query_sim_sentence/<q1>/<n>', methods=['GET'])
def query_sim_sentence(q1,n):
    global sentences_dict, vocab,database
    
    if len(sentences_dict)==0:
        return json.dumps({"info":"匹配文本库没有初始化语料"}, ensure_ascii=False)
    n=int(n)
    test_arr, query_arr = data_input.get_test_bert_singlequery([q1], vocab)
    q_emb = model.predict_embedding(test_arr)
    q_emb=np.reshape(q_emb[0],[-1,768])
    
    l1_distance=np.sum(np.abs(q_emb-database),axis=1)
    arg_index=l1_distance.argsort()[:n]
    rets=[int(i) for i in arg_index]
    
    index_info=[]
    for i in rets:
        q2=sentences_dict[i]
        test_arr2, query_arr2  =data_input.get_query_bert(q1,q2[1], vocab)
        label, prob = model.predict_nomerge(test_arr2)
        # index_info.append((q2[0],q2[1],float(prob[0])))
        index_info.append({"speech_term_id":q2[0], "speech_term_content": q2[1], "sim_score": float(prob[0])})
    # index_info=list(sorted(index_info,key=lambda x:x[2]))[::-1]
    index_info = list(sorted(index_info, key=lambda x: x["sim_score"]))[::-1]
    rets={"q1":q1,"n_recall":index_info}
    return json.dumps({"info":"返回成功","data":rets}, ensure_ascii=False)



def query_bm25_recall(q1,ids,n):
    global esobj
    ids=[int(e) for e in ids.split(",")]
    n=int(n)
    body={
      "query": {
        "bool": {
          "must": [
            {
              "terms": {
                "id": ids
              }
            },
            {
              "match": {
                "content": q1
              }
            }
            ]
        }
      },
      "size":n
    }
    rets=esobj.query("speech_term",body)
    jac_sim=[]
    docs=[]
    dids=[]
    for ele in rets:
        dids.append(ele["id"])
        q2=ele["content"]
        docs.append(q2)
        jac_sim.append(jaccard_similarity(q1,q2,False))
    return dids, docs,jac_sim

@app.route('/query_jaccard_sim_bm25_recall/<q>/<ids>/<n>', methods=['GET'])
def query_jaccard_sim_bm25_recall(q,ids,n):
    
    n=int(n)
    dids, docs,jac_sim=query_bm25_recall(q,ids,n)
    
    scores=list(sorted(zip(dids,docs,jac_sim),key=lambda x:x[2],reverse=True))
    ret=[]
    for item in scores :
        ret.append({"speech_term_id":int(item[0]), "speech_term_content": item[1], "jaccard_score": float(item[2])})
    ret={"q1":q,"n_recall":ret}
    
    return json.dumps({"info":"返回成功","data":ret}, ensure_ascii=False)

@app.route('/query_sim_sentence_bm25_recall/<q>/<ids>/<n>', methods=['GET'])
def query_sim_sentence_bm25_recall(q,ids,n):
    global vocab
    n=int(n)
    dids, docs,jac_sim=query_bm25_recall(q,ids,n)
    doc_doc=[]
    for e in docs:
        doc_doc.append([q,e])
    test_arr2, query_arr2  =data_input.get_test_bert_by_arr(doc_doc, vocab)
    _, prob = model.predict_nomerge(test_arr2)
    docs_score=list(sorted(zip(dids,docs,prob,jac_sim),key=lambda x:x[2],reverse=True))
    ret=[]
    
    for item in docs_score :
        ret.append({"speech_term_id":int(item[0]), "speech_term_content": item[1], "sim_score": float(item[2]), "jaccard_score": float(item[3])})
    ret={"q1":q,"n_recall":ret}
    ret=json.dumps({"info":"返回成功","data":ret}, ensure_ascii=False)
    return ret
    
        



def jaccard_similarity(s1, s2,lac):

    # 将字中间加入空格
    if lac:
        s1, s2 = lac_cut(s1), lac_cut(s2)
    else:
        s1, s2 = cut(s1), cut(s2)
        
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator




#读取数据库文本料初始化,创建好模型输出文本向量矩阵以及bm25模型
def init_source_data():
    cli=SqlClient()
    try:
        print("加载数据库数据。。。。。。。。。。。。。。。。。。。。。。")
        qa_df=[]
        dids=[]
        docs=[]
        sql="select id,content from test.speech_term "
        for i,e in enumerate(cli.query(sql)):
            did,doc=e
            dids.append(did)
            docs.append(doc)
            qa_df.append({"did":did,"sim":doc})
        print("加载数据库数据，并生成向量存储在内存，花费时间较长，请耐心等待。。。。")
        sentence_insert_into_database(dids,docs)
        print("创建Bm25Recall")
        bm25model = Bm25Recall(qa_df)
    finally:
        cli.close()
    return bm25model
    
    



@app.route('/home/', methods=['GET'])
def home():
    return "hello"

if __name__ == '__main__':
    lac = LAC(mode='seg')
    cfg_path = "./configs/config_bert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    model = SiamenseBert(cfg)
    # model.restore_session(cfg["checkpoint_dir"])

    sentences_dict=dict()
    database=np.array([])
    
    esobj=ESClient()
    
    
    # bm25model=init_source_data()
    app.run("0.0.0.0",80)

    
    
    #http://127.0.0.1/query_sim/lac/你好吗/你好呀
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
    
    
    query_sim_sentence_bm25_recall("怎么样学习好物理","1,2",5)

    query_bm25_recall("怎么样学习好物理","1,2",5)
    
