# 导入所需的模块
import sys
from django.forms.models import model_to_dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import datetime
from apps import models
from hand_realtime_classify import hand_realtime_score
import pymysql.cursors

# db = pymysql.connect(host='localhost',user='root',password='123456',database='rehealth')
#     # 使用 cursor() 方法创建一个游标对象 cursor
# cursor = db.cursor()
def data_patient(name):
    #条件条范围筛选
    # sql="select * from patient where name = '%s'"%name
    # cursor.execute(sql)
    # #这是查询表中所有的数据
    # x=cursor.fetchall()
    x = models.Patient.objects.filter(name='李四')
    context = model_to_dict(x[0])
    context['age'] = int(context['age'])
    context['medical_history'] = eval(context['medical_history'])
    context['training_plan'] = eval(context['training_plan'])
    context['action_list'] = context['training_plan'][:-1]
    context['cycle_num'] = int(context['training_plan'][-1])
    # for i in x:
    #     temp = model_to_dict(i)
    #     print(temp)
        # context[temp] = temp
    # print(type(x))
    # for (Name, pinyin,age,gender,password,medical_history,training_plan,
    #      training_result,classify_dataset,classify_model,score_dataset,doc_name) in x:
    #     age=int(age)
    #     medical_history=eval(medical_history)
    #     training_plan=eval(training_plan)
    #     action_list=training_plan[:-1]
    #     cycle_num=int(training_plan[-1])

    # db.close()
        
    # patient=dict(name=Name,pinyin=pinyin,age=age,gender=gender,last_time=last_time,
    #          password=password,medical_history=medical_history,action_list=action_list,
    #          cycle_num=cycle_num,training_result=training_result,classify_dataset=classify_dataset,
    #          classify_model=classify_model,score_dataset=score_dataset)
    # print(type(context))
    return context

def run_rehab(action_list,cycle_num,classify_model,score_dataset):
    score_list, lable, qualified_num ,standard_score= hand_realtime_score(standard_score=70, model_name='CNN',
                        cycle_num=cycle_num, action_list=action_list, classify_model=classify_model, score_dataset=score_dataset)
    qualified_rate = qualified_num / len(action_list)
    now = datetime.datetime.now()
    time_str = now.strftime('%Y_%m_%d_%H_%M_%S')
        # 在run方法中，把几个值封装成一个字典
    result = {'score_list': score_list, 'lable': lable, 'num': len(score_list),
                'qualified_rate': qualified_rate, 'standard_score': standard_score, 'time_str': time_str,'feedback':''}
    return result

def save_score(result,pinyin,training_result):
    # 使用动作名和时间作为文件名
    file_name = pinyin + '_' + result['time_str'] + '.pkl'
    # 拼接文件的路径，把log文件夹作为第一个参数
    file_path = os.path.join(training_result, file_name)
    data=dict(result=result,feedback='')
    # 使用pickle.dump方法把字典保存到文件中
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def plot_data(data,action_list,cycle_num):
    with open (data,'rb') as f:
        data=pickle.load(f)
    try:
        result=data['result']
    except:
        result=data
    # 获取result中的数据
    score_list = result['score_list']
    lable = result['lable']
    qualified_rate = result['qualified_rate']
    standard_score = result['standard_score']
    # time_str = result['time_str']
    plot1_x=lable
    plot1_y=score_list
    plot2_x=action_list
    plot2_y=[]
    print(score_list)
    print(np.sum(score_list[1::len(action_list)]))
    for i in range(len(action_list)-1):
        plot2_y.append(np.sum(score_list[i::len(action_list)])/cycle_num)

    plot2_y.append(np.sum(score_list[len(action_list)-1::len(action_list)])/(cycle_num-1))

    return plot1_x,plot1_y,plot2_x,plot2_y,qualified_rate,standard_score

def get_result_file(training_result):
    file_list=os.listdir(training_result)
    file_list.sort()
    file_list_path=[os.path.join(training_result,i) for i in file_list]
    return file_list,file_list_path

# patient=data_patient('李四')
# result=run_classify(patient['action_list'],patient['cycle_num'],patient['classify_model'],patient['score_dataset'])
# save_score(result,patient['pinyin'],patient['training_result'])
# plot1_x,plot1_y,plot2_x,plot2_y,qualified_rate,standard_score=plot_data('patients/李四,Lisi/result/Lisi_2023_07_12_15_26_13.pkl',['三','五','六','七','八','九'],3)
# print(plot1_x,plot1_y,plot2_x,plot2_y,qualified_rate,standard_score)
