import pickle
import numpy as np
import pandas as pd
# A={'U_ID':123,'姓名':'张三','拼音':'Zhangsan','gender':'男','age':50,'病历':'x','影像检查':'y','诊断':'z','训练方案':'Zhangsan.npy'}
# B={'U_ID':124,'姓名':'李四','拼音':'Lisi','gender':'女','age':53,'病历':'x','影像检查':'y','诊断':'z','训练方案':'Lisi.npy'}
# with open("patients/"+A['姓名']+','+A['拼音']+"/"+A['拼音']+".pkl", "wb") as tf:
#     pickle.dump(A,tf)
# with open("patients/"+B['姓名']+','+B['拼音']+"/"+B['拼音']+".pkl", "wb") as tf:
#     pickle.dump(B,tf)
# Zhangsan=['二','三','四','五','六','七',3]
# Lisi=['四','五','六','七','八','九',4]
# # print(Zhangsan,Lisi)
# np.save("patients/"+A['姓名']+','+A['拼音']+"/"+A['拼音']+".npy",Zhangsan,allow_pickle=True)
# np.save("patients/"+B['姓名']+','+B['拼音']+"/"+B['拼音']+".npy",Lisi,allow_pickle=True)
# motion_dict_pkl = {'零':0, '一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9}
# with open("zero_nine.pkl", "wb") as tf:
#     pickle.dump(motion_dict_pkl,tf)
motion_dict_np = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九','十','伸展','勾拳','握拳','7字型','直拳']
np.save('public/zero_nine.npy', motion_dict_np)
a=np.load('public/zero_nine.npy',allow_pickle=True).tolist()
print(a)
# old_actions=motion_dict_np.copy()
# old_score_actions=motion_dict_np.copy()
# np.save('zero_nine.npy', motion_dict_np)
# a=np.load('patients\李四,Lisi\Lisi.npy',allow_pickle=True)
# print(a)
with open('public/zero_nine.pkl','rb') as f:
    Zhangsan=pickle.load(f)
print(Zhangsan)