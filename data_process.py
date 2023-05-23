# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:53:27 2023

@author: 98608
"""

import pandas as pd 

df=pd.read_csv("./data/query_doc_label.txt",sep="\t")
train = df.sample(frac=0.8)

val_test=df[~df.index.isin(train.index)]

val = val_test.sample(frac=0.5)

test=val_test[~val_test.index.isin(val.index)]


train2=pd.read_csv("./data/train.csv",sep="\t")
train2.rename({"query1":"query","query2":"doc","label":"label"},inplace=True,axis=1)


val2=pd.read_csv("./data/dev.csv",sep="\t")
val2.rename({"query1":"query","query2":"doc","label":"label"},inplace=True,axis=1)


test2=pd.read_csv("./data/test.csv",sep="\t")
test2.rename({"query1":"query","query2":"doc","label":"label"},inplace=True,axis=1)



train.to_csv("./data/train_wentian.csv",index=False,sep="\t")
val.to_csv("./data/dev_wentian.csv",index=False,sep="\t")
test.to_csv("./data/test_wentian.csv",index=False,sep="\t")


pd.concat([train,train2]).to_csv("./data/train_merge.csv",index=False,sep="\t")
pd.concat([val,val2]).to_csv("./data/dev_merge.csv",index=False,sep="\t")
pd.concat([test,test2]).to_csv("./data/test_merge.csv",index=False,sep="\t")