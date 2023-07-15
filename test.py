# import torch
# print(torch.cuda.is_available())
# print(torch.__version__)
import pymysql.cursors
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime
# # #插入数据：有两种方法，第一种是直接写sql语句插入，第二种是通过pd.dataframe传入值
# # #方法一
# connection=pymysql.connect(host='localhost',port=3306,user='root',password='root',database='test',charset='utf8',cursorclass=pymysql.cursors.DictCursor)
# cursor = connection.cursor()
# cursor.execute( "insert into test2 (user_id,user_name) values (%s,%s)" %(3,"'banana'"))
# connection.commit()
# cursor.close()
# connection.close()
#方法二
engine = create_engine("mysql+pymysql://root:root@localhost:3306/rehabassist?charset=utf8")
df = pd.DataFrame(columns=['name', 'pinyin','age','gender','password','medical_history','training_plan','training_result','classify_dataset','classify_model','score_dataset'],
                  data=[['张三','Zhangsan','50','男','123456',
                         "[['%s','xxx'],['%s','yyy']]"%(str(datetime.date(year=2020,month=8,day=31)),str(datetime.date(year=2021,month=9,day=13))),
                         "['一','三','五','六','七','3']","patients/张三,Zhangsan/result/",
                         'pose_data\Zhangsan.csv','CNN_model\Zhangsan.pth','score_sample\Zhangsan.csv'],
                         ['李四','Lisi','53','女','123456',
                         "[['%s','xxx'],['%s','yyy']]"%(str(datetime.date(year=2020,month=8,day=31)),str(datetime.date(year=2021,month=9,day=13))),
                         "['三','五','六','七','八','九','3']","patients/李四,Lisi/result/",
                         'pose_data\Lisi.csv','CNN_model\Lisi.pth','score_sample\Lisi.csv']
                         ])
df.to_sql(name="patient",con=engine,if_exists='replace',index=False,index_label=False)
df = pd.DataFrame(columns=['name', 'pinyin','password','patient_name'],
                  data=[['王五','Wangwu','123456',"['张三','李四']"],
                         ['赵六','Zhaoliu','123456',"['张三','李四']"]])
df.to_sql(name="doctor",con=engine,if_exists='replace',index=False,index_label=False)
# a=np.array([[0,1,2,3,4,5,6,7,8,9],['零','一','二','三','四','五','六','七','八','九']])
# np.save('zero_nine.npy',a)
# a='[[0,1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,10,11]]'
# b=eval(a)
# print(b[0][1])
# c=str(b)
# print(c)