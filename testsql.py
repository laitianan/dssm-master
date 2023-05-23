import paddlehub as hub
import pymysql
#打开数据库连接
conn = pymysql.connect(host='cloudbility.blissmall.net',port=9015,user = "uesrk83",password = "KgZ3VV8FedS28wpNk8kQvkCf3wz09XiI",db = "test")


cur = conn.cursor()   # 数据库操作符 游标

dataset = hub.dataset.LCQMC()

k=0

for i,e in enumerate(dataset.test_examples):
    texta=e.text_a
    textb=e.text_b
    strs1='insert into speech_term(content,words) values ("%s","{}")'%(texta)
    
    strs2='insert into speech_term(content,words) values ("%s","{}")'%(textb)
    try:
        
        cur.execute(strs1)
        cur.execute(strs2)
    except Exception  as  e:
        k+=1
        pass
    print(i,k)
    if i%100==0:conn.commit()
    
# strs="delete from speech_term"
# cur.execute(strs)
conn.commit()  #提交数据
cur.close()    #关闭游标
conn.close()   #断开数据库,释放资源     
