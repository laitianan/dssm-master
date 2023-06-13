# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:00:37 2023

@author: 98608
"""

import re 
import numpy as np 

a='远乔国际大厦 (汉峪金谷A8 1201)'

res=re.findall(r"\(.+\)",'远乔国际大厦 (汉峪金谷A8 1201)')
print(res)

b=re.findall('\d+',a)
b=sorted(b,key=lambda x:len(x),reverse=True)



query_dict=dict()

with open("./data/data.txt","r",encoding='utf-8') as f:
    
    f.readline()
    i=0
    for line in f:
        line=line.split("\t")[-1][:-1]
        line=line.replace("\"","")
        if  not re.findall(r"1[3456789]\d{9}",line):
            query_dict[i]=line
            i+=1
            
keys=list(query_dict.keys())

n=len(keys)
import numpy as np 
from nlpcda import CharPositionExchange

def random_by_dl(ts):
    smw = CharPositionExchange(create_num=3, change_rate=0.3,char_gram=3,seed=1)
    rs=smw.replace(ts)

    return rs[1]

def random_text(text):
    
    
    n=len(text)
    
    rand=[]
    
    cut=2 
    
    while len(rand)<2:
        rd=np.random.randint(n)
        if ord("0")<=ord(text[rd]) and ord(text[rd])<=ord("9"):
            continue 
        
        if text[rd] in ["一","二","三","四","五","六","七","八","九","十",""]:
            continue 
        
        if ord("a")<=ord(text[rd]) and ord(text[rd])<=ord("z"):
            continue 
        
        if ord("A")<=ord(text[rd]) and ord(text[rd])<=ord("Z"):
            continue 
        
        if rd not in rand  :
            rand.append(rd)
    
    text=[e for e in text ]
    
    text[rand[0]],text[rand[1]]=text[rand[1]],text[rand[0]]
    
    return "".join(text) 
            

f_q=open("./data/train.txt","w",encoding='utf-8')
f_q.write("query\tdoc\tlabel\n")

for i,line in query_dict.items():

    numbers=re.findall('\d+',line)
    if i>=400000:
        break 
    if i%10000==0:
        print(i)
    
    if numbers:
        not_except=True
        numbers=sorted(numbers,key=lambda x:len(x),reverse=True)
        
        number=np.random.choice(numbers)
        l=len(number)
        
        num=10
        j=1
        while j<l:
            num=num*10
            j+=1
        try:
            
            random=np.random.randint(num//100,num)

            while str(random)==number:
                random=np.random.randint(num//100,num)
            
            random=str(random)
            
            query2=line.replace(number,random)
        except Exception as e:
            query2=line.replace(number,"")
            print(line)
            print(numbers,random)
            print(e)
            not_except=False

    
    if len(line)<=5:
        continue 
    
    if np.random.rand()<0.8:
        line2=line
        ch1=""
        ch2=""
        rep=["号楼","号","座","栋"]
        
        for ch in rep:
            if line2.count(ch):
                rep.remove(ch)
                ch1=ch
                ch2=np.random.choice(rep,1)[0]
                break
        if ch1!="":
            
            line2=line2.replace(ch1,ch2)

            # print("随机",line,"-----",line2)
        f_q.write(f"{line}\t{line2}\t1\n")
        
    else:
        f_q.write(f"{line}\t{line}\t1\n")
    
    if np.random.rand()<0.15:
        if np.random.rand()<0.5:
            line2=random_by_dl(line)
            # line2=random_text(line)
        else:
            
            line2=random_text(line)
        f_q.write(f"{line}\t{line2}\t1\n")
    
    ind=np.random.randint(n-1000)
    
    ran_keys=keys[ind:ind+1000]
    choice=np.random.choice(ran_keys,3)
 
    if i in choice:
        choice=[e for e in choice if e!=i]
    if numbers and not_except:
        f_q.write(f"{line}\t{query2}\t0\n")
        choice=choice[:-1]
    
    if not not_except:
        f_q.write(f"{line}\t{query2}\t1\n")
    
    for index in choice:
        query2=query_dict[index]
        
        f_q.write(f"{line}\t{query2}\t0\n")
            
        
with open("./data/data_nuber.txt","r",encoding='utf-8') as f:
    for line in f:
        line=line.replace("\"","").replace("\n","").split("\t")
        line1=line[0]
        line2=line[1]
        f_q.write(f"{line1}\t{line2}\t1\n")
f_q.close()



import pandas as pd 

df=pd.read_csv("./data/train.txt",sep="\t")
train = df.sample(frac=0.8)

val_test=df[~df.index.isin(train.index)]

val = val_test.sample(frac=0.5)

test=val_test[~val_test.index.isin(val.index)]

train.to_csv("./data/train_wentian.csv",index=False,sep="\t")
val.to_csv("./data/dev_wentian.csv",index=False,sep="\t")
test.to_csv("./data/test_wentian.csv",index=False,sep="\t")


