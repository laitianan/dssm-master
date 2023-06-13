# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:36:38 2023

@author: 98608
"""


import hashlib
 
# 创建MD5对象，可以直接传入要加密的数据
m = hashlib.md5('123456'.encode(encoding='utf-8'))
# m = hashlib.md5(b'123456') 与上面等价
print(hashlib.md5('123456'.encode(encoding='utf-8')).hexdigest())
print(m) 
print(m.hexdigest()) # 转化为16进制打印md5值
    
import pandas as pd 
columns=["order_id","receiver_province_name","receiver_city_name",\
         "receiver_district_name","receiver_address",\
             "receiver_full_address"]
df=pd.read_csv("./data/z.txt",sep="\t" ,header=None,error_bad_lines=False)
df.columns =columns
df=df.drop_duplicates(["order_id"],keep="first")
groups=df.groupby("receiver_district_name")

dfs=[]
for name ,g in groups:
    if g.shape[0]<=30:
        dfs.append(g)
    else:
        
        g["text"]=g.apply(lambda x:str(x["order_id"])+"####"+str(x["receiver_full_address"]), axis=1)
        
        g[["text"]].to_csv(f"./data/emb/query_emb_all_part{name}.txt",index=False,header=None)

df_s=pd.concat(dfs,axis=0)
df_s["text"]=df_s.apply(lambda x:str(x["order_id"])+"####"+str(x["receiver_full_address"]), axis=1)

name="小县城集合"
df_s[["text"]].to_csv(f"./data/emb/query_emb_all_part_{name}.txt",index=False,header=None)
    


# import collections 
# with open("./data/z.txt","r",encoding='utf8')as f:
#     length=collections.defaultdict(int)
#     for line in f:
        
        
#         l=line.replace("\n","").split("\t")
#         l=[e for e in l if e!=""]
#         l=len(l)
#         if l>6:
#             print(line)
#             break
#         length[l]=length[l]+1
