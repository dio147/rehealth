# 导入所需的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import pymysql.cursors

from patient_1 import *
from classify.CNN_hand import CNN_train
from get_hand_data import get_hand_data
from get_hand_score_data import get_hand_score_data

db = pymysql.connect(host='localhost',user='root',password='root',database='rehabassist')
    # 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
def data_doctor(name):
    #条件条范围筛选
    sql="select * from doctor where name = '%s'"%name
    cursor.execute(sql)
    #这是查询表中所有的数据
    x=cursor.fetchall()
    for (doctor_name, doctor_pinyin,doctor_password,patient_name) in x:
        patient_name=eval(patient_name)
    
    doctor=dict(name=doctor_name,pinyin=doctor_pinyin,password=doctor_password,patient_names=patient_name)

    return doctor

def check_classify_dataset():
    df=pd.read_csv('pose_data\zero_nine.csv',header=None)
    with open('public/zero_nine.pkl','rb') as f:
        action_dict=pickle.load(f)
    all_action_name=list(action_dict.keys())
    all_action_num=list(action_dict.values())
    data_count1=df.iloc[:,-1].value_counts()
    data_count=[data_count1[i] for i in all_action_num]
    return data_count,all_action_name,all_action_num

def check_score_dataset():
    df=pd.read_csv('score_sample\zero_nine.csv',header=None)
    with open('public/zero_nine.pkl','rb') as f:
        action_dict=pickle.load(f)
    all_action_name=list(action_dict.keys())
    all_action_num=list(action_dict.values())
    data_count1=df.iloc[:,-1].value_counts()
    data_count=[data_count1[i] for i in all_action_num]
    return data_count,all_action_name,all_action_num

def change_training_plan(patient_name,new_action_list,new_cycle_num):
    new_action_list.append(new_cycle_num)
    training_plan=str(new_action_list)
    sql="""UPDATE patient SET training_plan="%s" WHERE name='%s'"""%(training_plan,patient_name)
    cursor.execute(sql)
    db.commit()

def train_CNN(action_list,pinyin):
    # 从motion_data.csv文件中读取当前动作组的动作数
    classNum = len(action_list)
    print(classNum)
    # 调用CNN_train函数，传入动作数和动作组的英文名，开始训练CNN模型
    CNN_train(class_num=classNum, database=pinyin,action_list=action_list)

def add_classify_data(pose):
    with open('public/zero_nine.pkl','rb') as f:
        action_dict=pickle.load(f)
    all_action_name=list(action_dict.keys())
    all_action_num=list(action_dict.values())
    if pose in all_action_name:
        pose_index=all_action_name.index(pose)
    else:
        pose_index=len(all_action_name)
        all_action_name.append(pose)
        all_action_num.append(pose_index)
        action_dict[pose]=pose_index
        with open('public/zero_nine.pkl','wb') as f:
            pickle.dump(action_dict,f)
    get_hand_data(label=pose_index, database_path='pose_data/zero_nine.csv')
    return pose_index

def add_score_data(pose):
    with open('public/zero_nine.pkl','rb') as f:
        action_dict=pickle.load(f)
    all_action_name=list(action_dict.keys())
    all_action_num=list(action_dict.values())
    if pose in all_action_name:
        pose_index=all_action_name.index(pose)
    else:
        pose_index=len(all_action_name)
        all_action_name.append(pose)
        all_action_num.append(pose_index)
        action_dict[pose]=pose_index
        with open('public/zero_nine.pkl','wb') as f:
            pickle.dump(action_dict,f)
    get_hand_score_data(label=pose_index, database_path='score_sample/zero_nine.csv')
    return pose_index

doctor=data_doctor('王五')
patient=data_patient('李四')

plot_data('patients/李四,Lisi/result/Lisi_2023_07_11_21_43_41.pkl',patient['action_list'])
check_classify_dataset()
change_training_plan('李四',new_action_list=['一','二','三','四','六','七','八','九'],new_cycle_num=3)
patient=data_patient('李四')
print(patient['action_list'],patient['cycle_num'])
db.close()

